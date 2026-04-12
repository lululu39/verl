# verl PPO 源码调用链与公式实现对照

这份文档只基于当前代码库中的 PPO 主路径整理，重点回答两个问题：

1. verl 是如何一层层调用方法完成一次 PPO 训练 step 的。
2. PPO 的核心公式，在 verl 里分别落在哪些实现上。

为了避免把所有后端都混在一起，本文以当前最核心的公共控制流为主：

- 训练入口由 [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L300) 构建。
- 训练循环由 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1290) 的 RayPPOTrainer.fit 驱动。
- actor/critic 的真正更新，最终落到 FSDP worker 和 dp_actor/dp_critic；Megatron 路径结构类似，但本文优先展开更直观的 FSDP 实现。

## 1. 从入口到训练循环：PPO 是怎么跑起来的

### 1.1 最外层入口

主入口在 [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L300)。代码层级如下：

```text
main()
└─ run_ppo(config)
   └─ TaskRunner.run(config)
      ├─ add_actor_rollout_worker(config)
      ├─ add_critic_worker(config)
      ├─ add_ref_policy_worker(config)
      ├─ trainer = RayPPOTrainer(...)
      ├─ trainer.init_workers()
      └─ trainer.fit()
```

对应源码锚点：

- TaskRunner.run: [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L300)
- 构造 trainer: [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L376)
- 初始化 worker: [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L389)
- 启动训练: [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L392)

这一层做的事不是 PPO 数学本身，而是把 PPO 所需的执行角色装配起来：

- ActorRollout worker：负责 rollout、old log prob、actor update。
- Critic worker：负责 value 估计和 critic update。
- RefPolicy worker：当使用 KL 奖励或 KL loss 时，提供参考策略 log prob。
- Reward 侧：奖励模型或者规则奖励函数。

### 1.2 真正的 PPO 控制流在 RayPPOTrainer.fit

主循环在 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1290)。一轮 step 的关键调用顺序可以压缩成下面这棵树：

```text
RayPPOTrainer.fit()
├─ async_rollout_manager.generate_sequences(...)
├─ _compute_reward_colocate(...) / extract_reward(...)
├─ _compute_old_log_prob(batch)
├─ _compute_ref_log_prob(batch)          # 如果启用 reference policy
├─ _compute_values(batch)                # 如果启用 critic
├─ apply_kl_penalty(batch, ... )         # 如果 use_kl_in_reward=True
├─ compute_advantage(batch, ...)
├─ _update_critic(batch)                 # 如果启用 critic
└─ _update_actor(batch)
```

相关源码入口：

- KL 奖励整形: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L76)
- advantage 计算入口: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L136)
- 计算 values: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1141)
- 计算 ref log prob: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1158)
- 计算 old log prob: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1185)
- 更新 actor: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1217)
- 更新 critic: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1260)

fit 里的核心片段几乎就是 PPO 数据流本身：

```python
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

if self.use_rm and "rm_scores" not in batch.batch.keys():
    batch_reward = self._compute_reward_colocate(batch)
    batch = batch.union(batch_reward)

reward_tensor, reward_extra_infos_dict = extract_reward(batch)

old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
batch = batch.union(old_log_prob)

if self.use_reference_policy:
    ref_log_prob = self._compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)

if self.use_critic:
    values = self._compute_values(batch)
    batch = batch.union(values)

if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(batch, ...)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

batch = compute_advantage(batch, ...)

if self.use_critic:
    critic_output = self._update_critic(batch)

actor_output = self._update_actor(batch)
```

这段代码说明了一件很关键的事：verl 把 PPO 切成了两部分。

- driver 进程负责组织 rollout、奖励、KL 奖励整形、advantage 计算。
- worker 进程负责真正的神经网络前向和反向更新。

这也是为什么 fit 看起来像一个“调度器”，而不是一个大而全的 loss 函数。

## 2. 继续往下钻：actor/critic 更新是怎样一层层落地的

### 2.1 old log prob、ref log prob、value 的计算链

#### old log prob

```text
RayPPOTrainer._compute_old_log_prob
└─ actor_rollout_wg.compute_log_prob
   └─ FSDPWorker.compute_log_prob
      └─ DPActor.compute_log_prob
```

对应源码：

- trainer 入口: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1185)
- FSDP worker 包装层: [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1125)
- actor 前向求 log prob: [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py#L427)

这里得到的是 old_log_probs。它不是训练中“当前 mini-batch 更新后的策略”，而是这一轮 PPO update 的 proximal anchor，也就是 PPO 公式里的 $\pi_{old}$。

#### reference log prob

```text
RayPPOTrainer._compute_ref_log_prob
└─ ref_policy_wg.compute_ref_log_prob
   └─ FSDPWorker.compute_ref_log_prob
      └─ ref_policy.compute_log_prob
```

对应源码：

- trainer 入口: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1158)
- FSDP worker 层: [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1177)

reference policy 不参与 PPO clip 比值，它主要服务于两类约束：

- 奖励中的 KL penalty。
- loss 中的 KL regularization。

#### value 计算

```text
RayPPOTrainer._compute_values
└─ critic_wg.compute_values
   └─ FSDPWorker.compute_values
      └─ DPCritic.compute_values
```

对应源码：

- trainer 入口: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1141)
- FSDP worker 层: [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1678)
- critic 前向: [verl/workers/critic/dp_critic.py](verl/workers/critic/dp_critic.py#L153)

### 2.2 actor update 的调用链

```text
RayPPOTrainer._update_actor
└─ actor_rollout_wg.update_actor
   └─ FSDPWorker.update_actor
      └─ DPActor.update_policy
         └─ get_policy_loss_fn(loss_mode)
            └─ compute_policy_loss_vanilla  # 默认 vanilla PPO
```

对应源码：

- trainer 更新 actor: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1217)
- FSDP worker update_actor: [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1029)
- actor 真正训练循环: [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py#L513)
- PPO policy loss: [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1279)

DPActor.update_policy 里面做了三层循环：

```text
ppo_epochs
└─ mini_batches = data.split(ppo_mini_batch_size)
   └─ micro_batches = mini_batch.split(ppo_micro_batch_size_per_gpu)
      ├─ 前向得到 log_probs / entropys
      ├─ 计算 pg_loss
      ├─ 可选加 entropy bonus
      ├─ 可选加 KL loss
      └─ backward + optimizer.step()
```

这正对应了 PPO 常见的“同一批 rollout 数据上重复做多个 epoch 的小批量更新”。

### 2.3 critic update 的调用链

```text
RayPPOTrainer._update_critic
└─ critic_wg.update_critic
   └─ FSDPWorker.update_critic
      └─ DPCritic.update_critic
         └─ compute_value_loss
```

对应源码：

- trainer 更新 critic: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1260)
- FSDP worker update_critic: [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1698)
- critic 训练循环: [verl/workers/critic/dp_critic.py](verl/workers/critic/dp_critic.py#L192)
- value loss: [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L2084)

## 3. 公式到实现：verl 是怎样把 PPO 公式写进代码的

下面按 PPO 最核心的几个数学对象来对应源码。

### 3.1 KL 奖励整形

当使用 reference model 做 reward shaping 时，verl 使用的是：

$$
r_t = s_t - \beta \cdot \mathrm{KL}_t
$$

其中在默认 kl 模式下，代码里的 token 级 KL 近似是：

$$
\mathrm{KL}_t \approx \log \pi_{old}(a_t\mid s_t) - \log \pi_{ref}(a_t\mid s_t)
$$

这里 $s_t$ 是原始 token-level score，$\beta$ 来自 KL controller。

实现入口：

- KL 奖励整形: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L76)
- KL 具体计算: [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L2126)

对应代码：

```python
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]

    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld
    data.batch["token_level_rewards"] = token_level_rewards
```

以及 KL 的默认实现：

```python
def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob
```

解释：

- old_log_probs 是这一轮 PPO 更新的旧策略概率。
- ref_log_prob 是 reference policy 概率。
- token_level_rewards 是后续 advantage 计算真正使用的奖励。

如果没有启用 use_kl_in_reward，那么 fit 里会直接执行：

```python
batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```

也就是完全不做奖励层面的 KL 整形。

### 3.2 GAE：优势函数和回报

当 adv_estimator 选择 gae 时，verl 实现的是标准 GAE：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

$$
R_t = A_t + V(s_t)
$$

实现入口：

- advantage 总入口: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L136)
- GAE 具体实现: [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L216)

对应代码：

```python
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
```

这里有三个实现细节值得注意：

1. 它按 response token 倒序递推，所以和公式中的从后往前回传完全一致。
2. response_mask 会把 EOS 之后的 token 排除掉。
3. advantages 在最后做了 masked_whiten，也就是只在有效 response token 上标准化。

### 3.3 PPO clipped objective

PPO 的核心比值是：

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
= \exp\left(\log \pi_\theta(a_t\mid s_t) - \log \pi_{old}(a_t\mid s_t)\right)
$$

标准 clipped PPO 的目标是：

$$
L_t^{clip}(\theta) = \min\Big(r_t(\theta)A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\Big)
$$

由于代码里实现的是“最小化 loss”，所以等价写成：

$$
\ell_t^{clip}(\theta) = -L_t^{clip}(\theta)
$$

verl 的默认 vanilla policy loss 在标准 clip 之外，还带了 dual-clip 的分支，用于负优势样本：

$$
L_t^{dual-clip}(\theta) =
\begin{cases}
\min(r_tA_t, \mathrm{clip}(r_t)A_t), & A_t \ge 0 \\
\max(\min(r_tA_t, \mathrm{clip}(r_t)A_t), cA_t), & A_t < 0
\end{cases}
$$

实现位置：

- policy loss: [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1279)

对应代码：

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)

pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

这个实现和公式的对应关系是：

- ratio 对应 $r_t(\theta)$。
- pg_losses1 对应 $-r_tA_t$。
- pg_losses2 对应 $-\mathrm{clip}(r_t)A_t$。
- advantages < 0 时，再额外用 clip_ratio_c 做 dual-clip。

这个函数最后还会做统一聚合：

```python
pg_loss = agg_loss(
    loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
)
```

说明 verl 不只是逐 token 算 loss，还通过 response_mask 和 loss_agg_mode 控制最终在 token 维度和 batch 维度上如何聚合。

### 3.4 Actor 的总损失：PPO 项、熵奖励、可选 KL loss

actor 真正反向时，最终优化目标不是只含 policy clip。一种更准确的写法是：

$$
L_{actor} = L_{pg} - c_{ent} H + c_{kl} L_{kl}
$$

其中：

- $L_{pg}$ 是上面的 PPO policy loss。
- $H$ 是策略熵，系数是 entropy_coeff。
- $L_{kl}$ 是可选的 reference-policy KL regularization，只有 use_kl_loss=True 时启用。

在 legacy FSDP 路径里，这段逻辑直接写在 [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py#L513)。对应代码：

```python
pg_loss, pg_metrics = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    config=self.config,
    rollout_is_weights=rollout_is_weights,
)

policy_loss = pg_loss
if calculate_entropy and entropy is not None:
    entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    if entropy_coeff != 0:
        policy_loss -= entropy_agg * entropy_coeff

if self.config.use_kl_loss:
    ref_log_prob = model_inputs["ref_log_prob"]
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

如果你看的是新 model engine 的统一 loss 封装，那么对应入口在：

- PPO loss wrapper: [verl/workers/utils/losses.py](verl/workers/utils/losses.py#L58)

两条路径虽然包装层不同，但最后都收敛到同一套 core_algos 里的 PPO 核心公式。

### 3.5 Critic 的 clipped value loss

critic 对应的是 PPO 里的 value clipping：

$$
V_t^{clip}(\theta) = \mathrm{clip}\left(V_\theta(s_t), V_{old}(s_t)-\epsilon_v, V_{old}(s_t)+\epsilon_v\right)
$$

$$
L_t^{vf}(\theta) = \frac{1}{2}\max\left((V_\theta(s_t)-R_t)^2, (V_t^{clip}(\theta)-R_t)^2\right)
$$

实现位置：

- critic update 主体: [verl/workers/critic/dp_critic.py](verl/workers/critic/dp_critic.py#L192)
- value loss: [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L2084)

对应代码：

```python
def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

这里：

- vpreds 是当前 critic 的 $V_\theta(s_t)$。
- values 是 rollout 阶段保存下来的旧 value，也就是 $V_{old}(s_t)$。
- returns 就是前面 advantage 计算后得到的 $R_t$。

critic update 调用这段 loss 的方式是：

```python
vpreds = self._forward_micro_batch(model_inputs)
vf_loss, vf_clipfrac = core_algos.compute_value_loss(
    vpreds=vpreds,
    values=values,
    returns=returns,
    response_mask=response_mask,
    cliprange_value=self.config.cliprange_value,
    loss_agg_mode=self.config.loss_agg_mode,
)
```

## 4. 把整条 PPO 主路径串起来

如果把一次 PPO step 压成一句话，可以写成：

1. 用当前 actor rollout 出 response。
2. 计算 reward / rm score。
3. 重新计算 old_log_probs，必要时再算 ref_log_prob 与 values。
4. 用 reference policy 做 KL 奖励整形，得到 token_level_rewards。
5. 用 GAE 把 token_level_rewards 和 values 变成 advantages 与 returns。
6. critic 用 returns 做 clipped value regression。
7. actor 用 old_log_probs、advantages 和当前 log_probs 做 clipped PPO update。
8. 可选再叠加 entropy bonus 与 KL loss。

它在 verl 里的完整代码层级，可以概括为：

```text
main_ppo.py
└─ TaskRunner.run
   └─ RayPPOTrainer.fit
      ├─ rollout.generate_sequences
      ├─ reward / rm
      ├─ actor.compute_log_prob          -> old_log_probs
      ├─ ref_policy.compute_log_prob     -> ref_log_prob
      ├─ critic.compute_values           -> values
      ├─ apply_kl_penalty                -> token_level_rewards
      ├─ compute_gae_advantage_return    -> advantages, returns
      ├─ critic.update_critic            -> compute_value_loss
      └─ actor.update_policy             -> compute_policy_loss_vanilla
```

## 5. 建议的源码阅读顺序

如果你想顺着代码自己继续往下看，建议按这个顺序：

1. [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L300)
2. [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1290)
3. [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L76)
4. [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L136)
5. [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1029)
6. [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py#L513)
7. [verl/workers/critic/dp_critic.py](verl/workers/critic/dp_critic.py#L192)
8. [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1279)
9. [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L2084)

按这个顺序读，基本就能把“调度层 -> worker 层 -> 核心 PPO 数学实现”完整串起来。