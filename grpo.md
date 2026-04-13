# verl GRPO 源码调用链与公式实现对照

这份文档只基于当前代码库中的标准 RL 主路径整理，目标是回答两个问题：
= L_{clip}(\theta; A^{GRPO})
1. verl 是如何层层调用方法完成一次 GRPO 训练 step 的。
2. GRPO 的核心公式，在 verl 里分别落在哪些实现上。

为了把控制流讲清楚，本文只展开当前最核心的主路径：

- 训练入口： [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L300-L392)
- 主训练循环： [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1348-L1575)
- actor 真正更新： [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1029-L1069) -> [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py#L508-L673)
- GRPO advantage 核心实现： [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L267-L333)

先给一个结论，后面所有细节都围绕它展开：

- verl 没有单独的 GRPO trainer。
- GRPO 复用 PPO 的主训练器 `main_ppo -> RayPPOTrainer.fit`。
- GRPO 与 PPO 的主要差异，不在最外层调度，而在：
  - `algorithm.adv_estimator=grpo`
  - `rollout.n > 1`，同一 prompt 采多条响应形成 group
  - 通常不启用 critic
  - 常见配置下使用 actor 侧 `KL loss`，而不是 reward 侧 `KL penalty`

一个最典型的 GRPO 启动脚本就是 [examples/grpo_trainer/run_qwen3-8b.sh](examples/grpo_trainer/run_qwen3-8b.sh#L1-L40)：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=False
```

这四个配置基本就定义了 verl 里的“标准 GRPO”运行形态。

## 1. 从入口看：GRPO 如何挂到 PPO 主干上

### 1.1 训练入口仍然是 main_ppo

主入口在 [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py#L300-L392)。调用层级可以压缩成：

```text
main_ppo
└─ TaskRunner.run(config)
   ├─ add_actor_rollout_worker(config)
   ├─ add_critic_worker(config)
   ├─ add_ref_policy_worker(config)
   ├─ trainer = RayPPOTrainer(...)
   ├─ trainer.init_workers()
   └─ trainer.fit()
```

对应源码：

```python
actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
self.add_critic_worker(config)
self.add_ref_policy_worker(config, actor_rollout_cls)

trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    processor=processor,
    role_worker_mapping=self.role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    ray_worker_group_cls=ray_worker_group_cls,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    collate_fn=collate_fn,
    train_sampler=train_sampler,
)
trainer.init_workers()
trainer.fit()
```

这层本身不做 GRPO 数学计算，只负责把训练所需角色装配好。

### 1.2 为什么说“GRPO 只是 PPO 主干上的一个分支配置”

是否需要 critic、是否需要 ref policy，不是由“trainer 类名”决定，而是由配置决定。

看 [verl/trainer/ppo/utils.py](verl/trainer/ppo/utils.py#L75-L108)：

```python
def need_reference_policy(config: DictConfig) -> bool:
    return config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss


def need_critic(config: DictConfig) -> bool:
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False
```

这段代码说明：

- 当 `adv_estimator=grpo` 时，默认不会像 GAE/PPO 那样自动需要 critic。
- 当 `use_kl_loss=True` 或 `use_kl_in_reward=True` 时，才会拉起 reference policy worker。

所以在 verl 里，GRPO 的“算法身份”主要是由配置和优势函数实现决定的，而不是由单独 trainer 类决定的。

## 2. 一次 GRPO step 的代码层级分析

## 2.1 总调用链

一轮标准 GRPO step 可以压缩成下面这棵树：

```text
TaskRunner.run
└─ RayPPOTrainer.fit
   ├─ batch.non_tensor_batch["uid"] = uuid per prompt
   ├─ gen_batch_output = gen_batch.repeat(rollout.n, interleave=True)
   ├─ async_rollout_manager.generate_sequences(...)
   ├─ batch = batch.repeat(rollout.n, interleave=True)
   ├─ batch = batch.union(gen_batch_output)
   ├─ _compute_old_log_prob(batch)
   ├─ _compute_ref_log_prob(batch)              # 常见 GRPO 会走这里
   ├─ token_level_rewards = token_level_scores  # 当 use_kl_in_reward=False
   ├─ compute_advantage(..., adv_estimator=grpo)
   │  └─ core_algos.compute_grpo_outcome_advantage(...)
   └─ _update_actor(batch)
      └─ FSDPWorker.update_actor
         └─ DPActor.update_policy
            ├─ forward current policy log_prob
            ├─ compute_policy_loss_vanilla
            ├─ kl_penalty(log_prob, ref_log_prob, ...)
            └─ backward + optimizer.step()
```

这里最重要的事实有两个：

- group 的构造发生在 rollout 之前和 rollout 之后的 `repeat(interleave=True)`。
- GRPO 的核心差异发生在 `compute_advantage(... grpo ...)` 和 actor loss，而不是 trainer 外壳。

## 2.2 group 是怎样在 batch 里形成的

这一跳非常关键。GRPO 需要“同一 prompt 的多条响应”组成一个组，verl 的做法是：

1. 先给每个 prompt 分配一个唯一 `uid`。
2. 再把待 rollout 的输入按 `rollout.n` 做 `repeat(interleave=True)`。
3. `DataProto.repeat(..., interleave=True)` 会同时复制 tensor 字段和 `non_tensor_batch["uid"]`。
4. rollout 输出回并后，同一 prompt 的 `n` 条响应天然共享同一个 `uid`。
5. 之后 GRPO advantage 直接按这个 `uid` 做组内归一化。

对应主循环代码 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1362-L1407)：

```python
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
)

gen_batch = self._get_gen_batch(batch)
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
)

gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

而 `repeat(interleave=True)` 的语义在 [verl/protocol.py](verl/protocol.py#L972-L1015)：

```python
if interleave:
    repeated_tensors = {
        key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
    }

...

for key, val in self.non_tensor_batch.items():
    if interleave:
        repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
```

也就是说，同一个 prompt 的 `uid` 会被重复 `n` 次。后面 `compute_grpo_outcome_advantage(... index=data.non_tensor_batch["uid"])` 正是依赖这个分组键。

## 2.3 rollout 是怎样变成多条响应的

在 HuggingFace rollout 路径里，多返回序列本质上就是把 prompt 复制成 `num_return_sequences` 份，然后生成 `bs * n` 条响应。

见 [verl/workers/rollout/hf_rollout.py](verl/workers/rollout/hf_rollout.py#L136-L168)：

```python
generated_batch_size = seq.size(0)  # bs * num_return_sequences

num_return_sequences = kwargs.get("num_return_sequences", 1)
if num_return_sequences > 1:
    position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

prompt = seq[:, :prompt_length]
response = seq[:, prompt_length:]
```

这和 trainer 侧的 `repeat(interleave=True)` 组合起来，完成了 GRPO 所需的 group rollout 数据布局。

## 2.4 old policy / ref policy / actor update 的调用链

在主循环里，生成完响应后，verl 会先补齐 old log prob 和 ref log prob，然后才算 advantage 和 actor loss。

主循环对应代码 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1490-L1575)：

```python
old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
batch = batch.union(old_log_prob)

if self.use_reference_policy:
    ref_log_prob = self._compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)

if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(...)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam,
    num_repeat=self.config.actor_rollout_ref.rollout.n,
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    config=self.config.algorithm,
)

actor_output = self._update_actor(batch)
```

再继续往下钻，actor 更新链路是：

```text
RayPPOTrainer._update_actor
└─ actor_rollout_wg.update_actor
   └─ FSDPWorker.update_actor
      └─ DPActor.update_policy
         └─ compute_policy_loss_vanilla
```

关键入口：

- [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L1029-L1069)
- [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py#L508-L673)
- [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1279-L1368)

另外，FSDP actor worker 会把 `ppo_mini_batch_size` 乘上 `rollout.n`，说明 actor update 是直接在“展开后的所有响应”上做的，而不是按原 prompt 数更新：

```python
if self._is_actor:
    self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
```

源码位置： [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L250-L267)

## 3. 公式到实现：GRPO 在 verl 里是怎么写出来的

下面只对齐标准 GRPO 的几条核心公式。

## 3.1 Group sampling

GRPO 首先要求：对同一个 prompt $x_i$，采样 $G$ 条响应。

公式上可以写成：

$$
o_{i,1}, o_{i,2}, \dots, o_{i,G} \sim \pi_{\theta_{old}}(\cdot \mid x_i)
$$

在 verl 里，这个 $G$ 就是 `actor_rollout_ref.rollout.n`。主实现不是一个单独的 “group sampler” 类，而是 trainer 和 rollout 两层一起完成：

- trainer 侧做 `repeat(interleave=True)`： [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1367-L1407)
- rollout 侧生成 `bs * n` 条输出： [verl/workers/rollout/hf_rollout.py](verl/workers/rollout/hf_rollout.py#L136-L168)

对应代码：

```python
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
)
...
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

这一点非常重要，因为后面的组内均值和标准差，都是沿着同一个 `uid` 聚合的。

## 3.2 组内相对优势：GRPO 的核心公式

标准 outcome-only GRPO 的组内归一化写法可以写成：

先把一条响应的 token-level reward 汇总成一个标量分数：

$$
r_i = \sum_t r_{i,t}
$$

然后按同组样本计算均值和标准差：

$$
\mu_g = \frac{1}{|g|}\sum_{i \in g} r_i, \qquad
\sigma_g = \operatorname{std}(\{r_i : i \in g\})
$$

原始 GRPO 优势为：

$$
A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}
$$

如果关闭 `norm_adv_by_std_in_grpo`，则退化成：

$$
A_i = r_i - \mu_g
$$

verl 的实现和这个公式是逐行对应的，见 [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L267-L333)：

```python
scores = token_level_rewards.sum(dim=-1)

id2score = defaultdict(list)
id2mean = {}
id2std = {}

with torch.no_grad():
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    scores = scores.unsqueeze(-1) * response_mask

return scores, scores
```

这里有三个实现细节很值得注意：

1. `scores = token_level_rewards.sum(dim=-1)` 说明当前实现是 outcome-only GRPO，把一条响应的 token reward 先压成一个标量。
2. `index=data.non_tensor_batch["uid"]` 说明组是按 prompt uid 划分的，不是按 mini-batch 位置硬切。
3. `return scores, scores` 说明在标准 GRPO 分支里，`returns` 直接等于 `advantages`，因为没有 critic value head 参与 bootstrapping。

这个分发入口在 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L136-L206)：

```python
elif adv_estimator == AdvantageEstimator.GRPO:
    grpo_calculation_mask = data.batch["response_mask"]
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

## 3.3 为什么 advantage 最后会变成 token 级矩阵

虽然上面的 $A_i$ 是“每条响应一个标量”，但 actor loss 是按 token log-prob 来算的，所以 verl 会把这个标量广播回 token 维度：

$$
A_{i,t} = A_i \cdot \mathbf{1}[t \text{ 是有效 response token}]
$$

对应代码还是刚才那行：

```python
scores = scores.unsqueeze(-1) * response_mask
```

这意味着在 verl 里，GRPO 的组相对优势是“先序列级归一化，再广播到 token 级做 policy gradient”。

## 3.4 clipped policy objective：GRPO 在 verl 里仍然走 PPO-style ratio clipping

对当前策略 $\pi_\theta$ 和 rollout 时的旧策略 $\pi_{old}$，verl 先计算重要性比值：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{old}(a_t \mid s_t)}
= \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{old}(a_t \mid s_t))
$$

然后使用 PPO 风格的 clipped surrogate：

$$
L_{pg,t}(\theta) = -\min\left(
\rho_t(\theta) A_t,
\operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
$$

verl 这里还带有 dual-clip 处理，所以当 $A_t < 0$ 时还会再和 $c$ 做一次下界裁剪。

完整实现见 [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1279-L1368)：

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)

pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(
    ratio, 1 - cliprange_low, 1 + cliprange_high
)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
pg_loss = agg_loss(
    loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
)
```

这说明一个很关键的点：verl 里的 GRPO 不是“另起炉灶的 policy loss”，而是“GRPO 优势 + PPO clipped actor objective”的组合。

## 3.5 KL regularization：标准 GRPO 示例通常把 KL 加到 actor loss 里

verl 里的 GRPO 示例一般采用：

$$
L_{actor} = L_{pg} + \beta \cdot L_{KL}
$$

而不是：

$$
r_t' = r_t - \beta \cdot \mathrm{KL}_t
$$

这点可以直接从 GRPO 示例脚本看出来：

- `actor_rollout_ref.actor.use_kl_loss=True`
- `algorithm.use_kl_in_reward=False`

= L_{clip}(\theta; A^{GRPO})

actor 侧的实现见 [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py#L614-L645)：

```python
policy_loss = pg_loss

if self.config.use_kl_loss:
    ref_log_prob = model_inputs["ref_log_prob"]
    kld = kl_penalty(
        logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
    )
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

其中 `kl_penalty` 的默认常见形式是 `low_var_kl`，实现见 [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L2126-L2179)：

```python
if kl_penalty in ("low_var_kl", "k3"):
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    return torch.clamp(kld, min=-10, max=10)
```

写成公式就是：

$$
L_{KL}^{(k3)} = \exp(\log \pi_{ref} - \log \pi_\theta)
 - (\log \pi_{ref} - \log \pi_\theta) - 1
$$

所以标准 GRPO 配置下，verl 的 actor loss 更准确地说是：

$$
L_{actor}
= L_{clip}(\theta; A^{GRPO})
+ \beta \cdot L_{KL}(\pi_\theta, \pi_{ref})
- \alpha \cdot H(\pi_\theta)
$$

其中最后的 entropy 项只有在 `entropy_coeff != 0` 时才会启用。

## 3.6 reward 侧 KL 与 loss 侧 KL 的关系

如果你把配置改成 `algorithm.use_kl_in_reward=True`，那就会先做：

$$
r_t' = r_t - \beta \cdot \mathrm{KL}_t
$$

实现入口在 [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L76-L98)，核心代码是：

```python
kld = core_algos.kl_penalty(
    data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
)
kld = kld * response_mask
beta = kl_ctrl.value

token_level_rewards = token_level_scores - beta * kld
data.batch["token_level_rewards"] = token_level_rewards
```

但标准 GRPO 例子通常明确关闭这一分支，转而使用 actor 侧 KL loss。也就是说，verl 支持两种 KL 注入位置，但常见 GRPO 配置选择的是后者。

## 3.7 loss 聚合：为什么 verl 的默认 GRPO 更像 token-level 实现

即使组优势是序列级标量，verl 的 policy loss 默认仍然是按 token 聚合。

默认 `loss_agg_mode=token-mean`，其公式可以写成：

$$
L = \frac{\sum_{i,t} m_{i,t} \cdot \ell_{i,t}}{\sum_{i,t} m_{i,t}}
$$

其中 $m_{i,t}$ 是 `response_mask`。

实现见 [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1138-L1173)：

```python
if loss_agg_mode == "token-mean":
    if batch_num_tokens is None:
        if dp_size > 1:
            raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
        batch_num_tokens = loss_mask.sum()
    loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
```

这也解释了为什么 verl 文档里会特别强调：

- 原始 GRPO 论文更接近 sample-level loss
- verl 默认更常用 `token-mean`

因为在长链路推理场景里，这个实现通常更稳。

## 4. 把整条 GRPO 主路径重新串起来

如果把上面的实现压缩成一句更“代码化”的表述，可以写成：

1. `main_ppo` 根据配置组装 actor / rollout / ref worker。 
2. `RayPPOTrainer.fit` 给每个 prompt 分配 `uid`。 
3. 输入 batch 按 `rollout.n` 做 interleaved repeat，同一 prompt 派生出同组的多条 rollout。 
4. rollout 生成完成后，trainer 计算 `old_log_probs` 和 `ref_log_prob`。 
5. 若标准 GRPO 配置关闭 reward-side KL，则 `token_level_rewards = token_level_scores`。 
6. `compute_grpo_outcome_advantage` 对每个 `uid` 组做“组内均值/标准差归一化”，并把标量优势广播回 token 维。 
7. `DPActor.update_policy` 用 PPO-style clipped ratio objective 计算 `pg_loss`。 
8. 若启用 `use_kl_loss=True`，则再加上 actor-side KL regularization。 
9. 聚合 loss，反向传播，完成一次 GRPO actor update。 

把它写成最短调用链就是：

```text
main_ppo
-> RayPPOTrainer.fit
-> repeat prompts by rollout.n while preserving uid
-> generate_sequences
-> compute_old_log_prob / compute_ref_log_prob
-> compute_grpo_outcome_advantage(uid-grouped)
-> DPActor.update_policy
-> compute_policy_loss_vanilla + kl_penalty
-> optimizer.step
```

## 5. 和 PPO 相比，verl 里的 GRPO 到底变了什么

最后给一个最实用的对比结论：

- 外层 trainer：没变，仍然是 PPO 主干。
- rollout 方式：从单响应变成 `rollout.n > 1` 的 grouped sampling。
- advantage：从 GAE/value-based 变成 `uid` 分组的 relative reward normalization。
- critic：标准 GRPO 配置下通常不启用。
- actor loss：仍然是 PPO-style clipped objective，但优势项换成了 GRPO advantage。
- KL：标准 GRPO 示例更倾向于放在 actor loss，而不是 reward shaping。

如果你接下来要继续顺着源码深挖，最值得继续看的文件顺序是：

1. [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py)
2. [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py)
3. [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py)
4. [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py)
5. [examples/grpo_trainer/run_qwen3-8b.sh](examples/grpo_trainer/run_qwen3-8b.sh)