# ICLR 2027 冲刺执行方案 v3.0

## One Policy, Many Budgets: Budget-Contingent Search for Agentic Workflows

> 版本：v3.0，2026-08-04
>
> 目标：在 6 周内形成一篇可严格证伪、预算公平、可复现的 ICLR 2027 投稿；“best paper”是研究野心，不是可承诺结果。
>
> 项目简称：HBWS（Hard-Budget Workflow Search）。

---

## 0. 先给结论：当前创新能否奔着 ICLR best paper 去？

### 0.1 判断

v2 的方向有 ICLR 投稿潜力，但按原稿还不足以支撑 best-paper 级主张。问题不在于“自动搜索 Workflow”不重要，而在于 v2 实际落下来的算法仍接近“进化搜索 + 预算阈值分支 + 成本报告”：

1. `budget_below:f` 只是搜索空间中的一个原语；v2 没有让“一个策略覆盖多个预算”进入候选评价、亲子信用分配和 archive 更新。
2. v2 的单一效用 `U(w)=score-cost+UCB` 与跨预算主张不一致，候选可能只在一个预算档投机。
3. “分别为每个预算搜索静态 Workflow”既被当作能力上界，又被要求与 HBWS 使用相同总搜索预算，两个科学问题混在了一起。
4. v2 错误写了“未检索到 AgentEvo”。AgentEvo 已于 2026 年发表，且直接研究成本感知、多阶段进化、性能—成本 archive，并在 MATH、HumanEval、MBPP 等高度重叠任务上实验；忽略它会成为明显的 Related Work 硬伤。
5. v2 只用最紧预算维度的剩余比例做分支，丢失了 token、调用、工具、时延、美元五维预算的不同可行性。
6. v2 的 H1 没有明确非劣界限，且把“两个预算档均非劣”和“至少一个预算档优越”混成一句；统计上不可预注册。
7. “减少人工设计负担”没有测量。自动搜索美元成本不能自动等同于人力减少；若不记录人工干预次数、种子数量和逐预算调参次数，就不能下这个结论。
8. 19,800 次执行的估算遗漏 `$0.15`、OOD、搜索 seed 产生的多个最终 policy 以及消融，成本模型不完整。

### 0.2 v3 的 best-paper 级加强点

v3 将贡献收束为一个相互依赖的三件套：

1. **问题设定**：搜索对象不是预算档对应的静态图，而是受限 DSL 中的 Budget-Contingent Workflow Policy；一个 policy 在运行时读取反馈、完整剩余预算向量和节点可预留性，决定继续、验证、投票、回退或停止。
2. **算法**：跨预算风险敏感目标、逐预算可行性门、跨预算亲子 credit、保守不确定性选择和多保真 racing 共同构成 HBWS；不是给旧搜索器加一个 `if budget`。
3. **评估协议**：Protocol A 测部署能力上界，Protocol B 测设计效率；同时报告部署成本、搜索成本、静态上界差距（budget-regret）和摊销临界点。

如果最终只证明“HBWS 与静态方法差不多”，这仍可能是一篇严谨的经验论文，但不是 best-paper 叙事。要保留 best-paper 竞争力，最终至少需要同时满足：

- 两个任务族、两个已见预算上均达到预注册非劣；
- 至少一个已见预算显著优于强静态对照，或显著降低相同质量下的部署成本；
- 未见 `$0.15` 预算不崩；
- 跨预算目标和 budget-conditioned 分支存在清晰的因果消融；
- 自动发现的机制可解释且能在成功/失败案例中复现；
- 搜索成本与摊销分析经得起复算。

---

## 1. Executive Summary

复杂 Agent Workflow 仍高度依赖人工决定节点、角色、拓扑、Prompt、验证、循环和终止条件。本项目不试图消灭人工 Harness，而是把分工改为：人定义安全、可审计的设计空间，Meta-Agent 在其中搜索具体 Workflow policy。

本文提出 **Budget-Contingent Agent Architecture Search**。给定任务分布、受限 Workflow DSL、少量人工种子和一组部署预算，HBWS 搜索一个跨预算执行的 policy。该 policy 以当前中间产物、合法反馈、归一化剩余预算向量、节点可预留掩码和循环状态为输入，动态选择下一节点或 END。主实验每个任务族单独搜索一个 policy；本文不声称同一个 policy 跨数学与代码通用。

HBWS 的核心不是预算阈值，而是跨预算训练与选择。每个候选在 `$0.10` 和 `$0.25` 上配对评估；以逐预算成功率 LCB、风险敏感 AUBPC、最差预算表现和跨预算 archive 贡献共同决定保留与变异 credit。任何一个预算档发生可归因于 policy 的硬预算违约，候选即不可行。多保真评估使用固定、按难度分层、严格嵌套的任务子集，只有统计上仍可能改善 archive 的候选晋级。

实验明确分成两套协议。Protocol A 允许 Static Evolution Search 为每个预算档获得独立、充分的搜索预算，用作部署能力上界；Protocol B 给 HBWS 和静态搜索相同总搜索预算，测量覆盖多个部署预算时的设计效率。两者均在相同执行模型、工具、DSL 原语、变异算子、任务切分、评估器和逐任务预算下运行。

最低任务设置为 MBPP+ → HumanEval+ OOD，以及 MATH 高难度四学科 → 两个未见学科 OOD。搜索预算为 `$0.10/$0.25`，`$0.15` 完全不参与搜索、阈值调节和候选选择。HBWS、Static Evolution Search 和 Random Search 各运行 3 个独立搜索 seed；最终候选各运行至少 3 个执行 seed。

---

## 2. 一句话论文主张

> We formulate agent-workflow design as the search for a single budget-contingent policy over a safe, auditable DSL, and introduce a risk-sensitive cross-budget evolutionary procedure that is evaluated against per-budget static search under hard per-instance budgets and fully accounted search costs.

这句话只主张本文做了什么，不提前声称实验胜出。得到结果后，只有通过预注册门槛才能增加 “matches or improves …” 等结果性表述。

---

## 3. 贡献、非贡献与相关工作

### 3.1 三个相互依赖的贡献

**贡献 1：Budget-Contingent Agent Architecture Search 问题。**

- 搜索一个跨预算 policy，而不是为每个预算分别选一个静态图；
- 联合设计拓扑、角色、Prompt、聚合器、条件边、循环和终止；
- 运行时使用合法反馈和完整预算状态改变控制流；
- 用相对 per-budget static upper envelope 的 budget-regret 衡量共享 policy 的代价。

**贡献 2：风险敏感的跨预算联合进化。**

- 候选必须在所有搜索预算上评估；
- 选择同时优化加权 AUBPC-LCB 与最差预算 LCB；
- 亲子 credit 要求“任何预算不显著退化、至少一个预算可靠改善”或产生可靠的跨预算 archive 贡献；
- 固定嵌套任务子集、置信界 racing 和逐阶段变异降低搜索成本。

**贡献 3：严格区分部署能力和设计效率的评估协议。**

- Protocol A 给静态方法每预算独立充分搜索的优势；
- Protocol B 匹配总搜索预算；
- 强制区分部署成本、搜索成本、账单成本、逻辑计算成本和摊销临界点；
- 逐任务硬预算通过调用前预留执行，而不是调用后才判失败。

### 3.2 明确不是创新的部分

下列组件可作为严谨工程或必要实验设计，但不得单独写成创新：

- 进化搜索、三阶段搜索、多保真评估、Successive Halving；
- Prompt 搜索、拓扑搜索、角色搜索、投票、循环、验证节点；
- Pareto/非支配 archive 本身；
- 单个 `remaining_budget < threshold` 条件；
- 成本账本、sandbox、DSL 白名单；
- 在 MATH、MBPP、HumanEval 上评测；
- 自动 Workflow 搜索或“减少人工设计”的一般愿景。

### 3.3 相关工作差异矩阵

下表只写可由论文公开描述支持的差异；投稿前需逐篇精读正文和最新版本，不能只依赖摘要。

| 工作 | 已公开的核心对象/机制 | 与 HBWS 的重叠 | HBWS 需要证明的差异 |
|---|---|---|---|
| GPTSwarm | 将语言 Agent 表示为可优化图，优化节点 Prompt 和边连接 | Prompt 与图结构联合优化 | HBWS 搜索受硬预算约束、运行时预算条件化的单 policy，并作跨预算联合评价 |
| ADAS | 以 meta-programming 自动发明代码定义 Agent | 自动设计完整 Agent 系统 | HBWS 使用受限声明式 DSL、任务级预留账本和跨预算目标，不执行 Meta-Agent 生成的任意 Workflow 代码 |
| AFlow | 通过 MCTS 和代码级修改搜索可执行 Workflow | 自动 Workflow 搜索、执行反馈 | HBWS 的核心变量是跨预算 policy；主对照不依赖不完全兼容的 AFlow 复现 |
| AgentSquare | 模块化设计空间、模块进化与重组、性能预测器 | 模块化搜索和搜索加速 | HBWS 联合搜索条件控制流并严格比较 per-budget static upper bound |
| EvoFlow | niching 进化一组异构、复杂度自适应 Workflow | 进化、工作流群体、成本效率 | 不把 EvoFlow 简化成“所有静态”；只主张 HBWS 显式优化一个跨部署预算的条件 policy |
| AgentEvo | 成本感知多阶段进化、压缩/交叉、性能—token cost 非支配 archive；MATH/HumanEval/MBPP 等 | 最接近：进化、成本和相同任务 | AgentEvo 公开目标是 Workflow 的性能—平均 token cost archive；HBWS 的区别必须由逐任务硬预算、单策略跨预算、跨预算联合 credit、Protocol A/B 和摊销分析建立 |
| BAVT | 用剩余资源比例条件化推理树节点选择 | 运行时预算条件化 | BAVT 作用于单次推理树的节点选择；HBWS 搜索完整 Workflow policy 的拓扑、Prompt、反馈分支、循环和终止 |
| Agent-UCT | 成本感知 UCT、配置前缀复用、缓存和可复现 Workflow 测试 | 搜索成本与复用 | Agent-UCT 重点是搜索器侧的评估效率；HBWS 同时研究被搜索对象本身的预算条件行为和跨预算部署 |

**对 AgentEvo 的处理。** 它必须进入正文 Related Work 和差异表。主受控基线仍是 Static Evolution Search，因为它可以与 HBWS 共享 DSL、模型、算子、任务、预算和代码路径。投稿前再次核验 AgentEvo 是否发布官方 artifact；若没有可直接复用 artifact 或协议无法配平，不做伪“复现”，只做方法级讨论和尽可能透明的适配实验。不能把未复现写成对方不存在。

**禁止主张。** “首次自动 Agent 设计”“全面替代人工 Harness”“已有方法都只搜索静态图”“首次预算自适应 Agent”“全局 Pareto-optimal”，以及任何未配平模型、工具、预算和评估器的性能结论。

---

## 4. 形式化问题与评价量

### 4.1 对象与边界

对任务族 (d) 单独搜索 policy π；主论文不要求一个 policy 同时处理代码和数学。令任务 (x\sim D_d)，部署预算 profile 为

\[
C_b=(C_b^{in}, C_b^{out}, C_b^{llm}, C_b^{tool}, C_b^{wall}, C_b^{usd}),
\]

其中 `$0.10/$0.25` 是 profile 名称和美元上限，其他维度也有独立硬帽。美元不是唯一预算；它由 token 单价、工具费等结算，但仍作为独立部署约束。

运行状态为

\[
s_t=(z_t,f_t,\rho_t,m_t,h_t),
\]

其中 (z_t) 是中间产物的有界摘要，(f_t) 是合法反馈，ρ_t 是各预算维度剩余量除以初始 cap 的向量，(m_t) 是对所有下一节点的 `reserve_feasible` 掩码，(h_t) 是循环计数与执行历史。policy π 根据 (s_t) 选择下一 DSL 节点、fallback 或 END。

policy 不读取 budget ID（如 `tight/loose`）或 `$0.10` 这样的离散标签；它只能读取连续剩余比例、可预留掩码和白名单反馈。这避免把两个训练预算记忆成两个静态图，并使 `$0.15` 测试有意义。

### 4.2 成功率和风险敏感跨预算目标

在搜索预算集合 (B_s=\{0.10,0.25\}) 上，对候选 π 计算逐预算成功率及其保守下界：

\[
L_b(\pi)=\operatorname{LCB}_{1-\alpha_s}[\Pr(Y=1\mid \pi,b)].
\]

由于候选是自适应产生和反复筛选的，搜索期的 LCB 使用对评估次数有效的 empirical-Bernstein confidence sequence（或 prereg 中冻结的等价 anytime-valid bound），而不是把普通 bootstrap CI 当成选择后推断。搜索 LCB 只是候选选择统计量；确认性 CI 一律来自完全独立的冻结测试集。

搜索阶段的离散 AUBPC 为

\[
\operatorname{AUBPC}_{LCB}(\pi)=\sum_{b\in B_s} w_bL_b(\pi),\quad
w_{0.10}=w_{0.25}=0.5.
\]

这里的 AUBPC 是归一化的两端点梯形积分：对 `[0.10,0.25]` 内作线性插值并除以区间长度后，恰为两个端点成功率的平均。`$0.15` 不进入搜索积分，只用于真正的未见预算检验。

仅用加权平均仍可能允许单预算投机，因此主排序分数预注册为

\[
J_{CB}(\pi)=\lambda\operatorname{AUBPC}_{LCB}(\pi)
 +(1-\lambda)\min_{b\in B_s}L_b(\pi),\quad \lambda=0.5.
\]

候选还必须满足：

1. 每个预算档的 policy-attributable violation rate 为 0；
2. 每档 (L_b(\pi)) 不低于预注册最低可部署门槛 (q_{d,b})；门槛在 pilot 后、正式搜索前根据 Direct/CoT 基线冻结；
3. 任何搜索阶段不得接触最终测试集或 `$0.15` 结果。

archive 保存经验非支配向量

\[
v(\pi)=\{L_b(\pi),-\bar c_b(\pi),-P95_b(\pi)\}_{b\in B_s},
\]

称为 **empirical cross-budget archive**，不称全局 Pareto frontier。

### 4.3 Budget-regret：共享 policy 距离每预算静态上界多远

令 (U_b^{static}) 为 Protocol A 中该预算档独立充分搜索得到的最好冻结静态候选的测试成功率。定义

\[
R_{budget}(\pi)=\sum_b w_b\max(0,U_b^{static}-S_b(\pi)),
\]

以及最坏档差距

\[
R_{max}(\pi)=\max_b(U_b^{static}-S_b(\pi)).
\]

它们只用于最终评价，不用于 HBWS 搜索，以免把静态基线的额外搜索信息泄漏给 HBWS。该量直接回答：用一个 policy 覆盖多预算到底损失了多少能力。

### 4.4 Cross-budget credit assignment

对 parent (p) 和 child (c)，在相同任务、相同执行 seed 上配对执行，计算

\[
\Delta_b=\operatorname{LCB}[S_b(c)-S_b(p)].
\]

预注册容忍度 ε=0.01。一次变异获得：

- **正 credit**：child 可行，且所有预算档 Δ_b≥−ε，同时至少一档 Δ_b>ε；或其保守 cross-budget hypervolume contribution 大于预注册最小值；
- **负 credit**：任一档 Δ_b<−ε，任一档发生 policy-attributable 预算违约，或结构无 END/无法静态验证；
- **中性 credit**：置信区间不足以区分。

credit 乘以 fidelity 权重 (1,2,4)，避免大量低保真噪声压过少量高保真证据。按“阶段 × 变异类型 × 任务族”维护 Beta-Bernoulli 成功统计，使用 Thompson sampling 调度下一变异类型；这只是进化搜索内部的算子调度，不另立第二种主搜索方法。

防止单档投机的四层机制是：逐档可行门、(J_{CB}) 的最差档项、亲子 credit 的无显著退化条件、跨预算 archive。消融必须逐一证明不是只有 budget branch 原语在起作用。

---

## 5. HBWS 系统设计

### 5.1 受限、可审计 Workflow DSL

节点白名单：

| 节点 | 输入/输出 | 可搜索参数 | 反馈权限 |
|---|---|---|---|
| `generate` | 问题 → 候选答案 | role、prompt_id、temperature、token cap | 无 gold |
| `refine` | 候选+反馈 → 新候选 | role、prompt_id、是否保留 incumbent | 仅合法反馈 |
| `vote` | 多候选 → 候选 | k≤5、聚合规则、并行/串行 | 无 gold |
| `oracle_verify` | 代码+可见测试 → 结构化反馈 | 测试子集、超时 | 仅代码任务；不得访问隐藏测试 |
| `heuristic_critic` | 数学解答 → 结构化反馈 | 独立重解/一致性/格式检查 | 不得读取 gold |
| `decompose` | 问题 → 子问题 | prompt_id、最大子问题数 | 无 gold |
| `aggregate` | 子结果 → 候选 | prompt_id、规则 | 无 gold |
| `branch` | 状态 → 下一节点 | 白名单谓词、阈值 | 只读状态 |
| `END` | incumbent → 最终输出 | fallback 规则 | 无 |

结构约束：节点数≤8；循环数≤2；每循环最多 3 次；vote k≤5；所有可达状态必须有 END 或静态可证明的有限循环退出；最长 LLM 调用数还受当前 budget profile 的调用硬帽限制。

条件谓词白名单：

- `feedback.kind == PASS/FAIL/UNCERTAIN`；
- `remaining_fraction(metric) < q`；
- `reserve_feasible(node_id)`；
- `consensus >= q`；
- `iteration_count < k`；
- 上述原子的 AND/OR，最多 3 个原子，禁止任意表达式和动态代码。

Meta-Agent 只能生成 JSON/AST patch，由 schema validator、图验证器和预算验证器解释；不得生成并执行 Python。代码题的候选解答是任务输出，必须在网络隔离 sandbox 中执行，不能与“执行 Meta-Agent 生成的 Workflow 代码”混淆。

### 5.2 反馈语义

**代码任务。** `oracle_verify` 只能使用题目明确暴露给 Agent 的可见样例或搜索集公开测试。HumanEval+/MBPP+ 隐藏/扩展测试只存在于外部最终评估层。若公开题面没有足够可见测试，必须从训练部分构造且固定可见测试，并保证不复制隐藏断言。

**数学任务。** `heuristic_critic` 只能使用独立重解一致性、步骤完整性、量纲/格式检查或模型自评。gold answer、官方解答和由 gold 派生的提示不进入 Workflow state。最终 exact-match/等价判分在外部评估层完成。

所有反馈写入带 provenance 的结构化记录：来源、时间、调用成本、可见性等级、是否允许影响控制流。

### 5.3 硬预算账本与预留

每个任务单独维护累计账本。调用节点 (n) 前计算基于 schema 的最坏可计费向量 ĉ(n,s_t)：最大输入上下文、最大输出 token、LLM/工具调用数、节点 deadline 和按冻结价表计算的美元上界。

```text
reserve(c_hat):
    if any(used + reserved + c_hat > cap): return REJECT
    reserved += c_hat
    return lease_id

settle(lease_id, actual):
    assert actual <= reserved_for_lease   # 超出视为运行时缺陷
    used += actual
    reserved -= reserved_for_lease
```

- 预留失败时节点不得调用，只能沿 `reserve_feasible == false` 的 fallback 或 END；
- vote 并行调用必须在启动前一次性预留全部并发调用；
- 完成后按实际 token/调用/工具费结算并返还差额；
- 墙钟时间使用每调用 deadline 和进程级 watchdog；远端 API 取消可能存在不可控尾延迟，需把“policy 违约”和“provider cancellation lag”分开报告；
- 价格表在 prereg commit 中保存日期、区域、部署 SKU 和截图/导出；论文中的美元成本按冻结价表重算，token 与调用数始终同时报告。

**预算安全命题（必须完成证明）。** 若每个节点的实际可计量资源不超过其预留、并发调用在启动前全部预留、所有路径只经账本启动节点，则对任意任务和任意 DSL policy，除明确隔离的外部取消延迟外，累计可计量资源不超过任务 cap。证明对成功 settle 次数作归纳即可，放附录。

### 5.4 三阶段进化

1. **Stage 1：骨架与拓扑。** 变异节点增删、边重连、条件边、有限循环、fallback 和 END；prompt 固定为注册表默认值。
2. **Stage 2：节点配置。** 搜索角色、投票数、聚合器、验证方式、保留 incumbent、token cap；拓扑只允许局部修复。
3. **Stage 3：局部 Prompt 优化。** 只允许注册表 patch；不改变数据可见性、工具权限和预算语义。

模板变异是主方法。LLM 生成变异属于时间允许项，因为它引入新的优化模型成本和安全变量，不能拖累主线。

### 5.5 多保真评估与不确定性选择

每个任务族预先按难度/类别分层生成固定嵌套集合：

- F1：24 题 × 1 执行 seed；
- F2：64 题 × 2 执行 seed，包含 F1；
- F3：完整 search-dev（目标 120 题）× 3 执行 seed，包含 F2。

实际题数必须在数据审计后冻结；若某层不足，保持 1:2.5:5 的近似比例，不临时挑题。晋级使用 Successive Halving 加置信界 racing：child 的乐观上界若已低于 archive 可比候选的保守下界且没有结构新颖性，则停止；否则晋级。F1/F2 已执行样本在高层复用，账单计入原发生阶段。

静态 DSL 验证失败不调用模型、成本为 0，但失败率必须报告。API 瞬时失败按所有方法统一的预注册重试规则处理；重试 token 和时间计入搜索成本。

### 5.6 伪代码

```text
HBWS(D_search, B={0.10, 0.25}, seeds, total_search_cap):
    initialize population from fixed human seeds + legal random seeds
    initialize empirical cross-budget archive A
    initialize mutation-credit tables Q

    for stage in [TOPOLOGY, CONFIGURATION, PROMPT]:
        while stage_budget_remaining:
            parent <- select(A, robust_score=J_CB, uncertainty, diversity)
            op <- ThompsonSample(Q[stage, task_family])
            child <- apply_declarative_mutation(parent, op)

            if not static_validate(child):
                assign_negative_credit(op, reason="invalid")
                continue

            for fidelity in [F1, F2, F3]:
                for b in B:
                    evaluate child and parent on identical tasks/seeds
                    enforce per-instance reserve ledger
                estimate L_b(child), paired Delta_b, cost, latency

                if any policy-attributable budget violation:
                    mark child infeasible; break
                if cannot_beat_archive_under_optimistic_bound(child):
                    break

            credit <- cross_budget_credit(Delta, feasibility,
                                          archive_contribution,
                                          fidelity)
            update Q with credit
            if deployable(child): update empirical archive A

    return epsilon-deduplicated cross-budget archive A
```

### 5.7 复杂度与成本上界

令第 (l) 个 fidelity 有 (N_l) 个候选、(m_l) 个任务、(r_l) 个执行 seed、(K=|B_s|=2)，每个任务的最大节点执行数为 (H\leq\sum_b C_b^{llm}) 的相应档限制。逻辑调用复杂度为

\[
O\left(\sum_l N_lm_lr_lKH\right).
\]

由于每次任务执行受美元 cap，候选评估的保守账单上界为

\[
C_{eval}\leq \sum_lN_lm_lr_l\sum_{b\in B_s}C_b^{usd},
\]

另加 optimizer 调用、工具/沙箱计算、失败重试。该上界很松；pilot 后用节点级 P50/P95 token 分布给出期望、P95 和绝对上界三列，不能只报均值。

---

## 6. Research Questions、假设与预注册统计

### 6.1 RQ1：自动设计有效性

在相同执行模型、工具、评估器、逐任务预算和数据切分下，HBWS 是否优于 Direct、CoT/自检、人工强 Workflow、Prompt-only、Topology-only、Random Search 和 Static Evolution Search？

**H1a（确认性）**：HBWS 的跨预算 AUBPC 高于人工强 Workflow 和 Random Search；以 paired bootstrap 的 Holm 校正 95% CI 判定。

**H1b（机制性）**：Full HBWS 优于去掉跨预算目标的版本；若 CI 不支持，则“跨预算联合搜索机制有效”这一贡献不成立。

### 6.2 RQ2：预算自适应性

**H2a（已见预算非劣）**：对 `$0.10` 和 `$0.25`，分别检验

\[
LCB_{one-sided,95\%}[S(HBWS,b)-S(Static_A,b)]\geq-\delta,
\quad \delta=0.03.
\]

两个预算档都通过才可声称非劣。非劣检验使用单侧区间；不能用“差异不显著”代替非劣。

H2a 是 intersection-union 结论：只有两个单档原假设都被拒绝才宣布整体非劣。H2b 的“至少一档优越”另作 Holm 校正，不能用 H2a 的单侧区间替代。

**H2b（至少一档优越）**：两个预注册已见预算中至少一个满足 Holm 校正后的双侧 95% CI 下界大于 0。只有 H2a 两档均通过后才正式解释 H2b。

**H2c（未见预算）**：`$0.15` 完全不参与搜索。HBWS 与两个静态迁移对照比较：（i）把 `$0.10` 搜出的 static policy 直接放到 `$0.15`；（ii）把 `$0.25` policy 在 `$0.15` 硬帽下执行；取二者中预先按 validation 规则选择者。不得为 `$0.15` 重新搜索或调阈值。主要报告差值与 CI；若希望做确认性非劣，同样使用 δ=3pt 并在 prereg 中冻结。

### 6.3 RQ3：搜索效率与摊销

**H3a（多保真）**：在三个搜索 seed 的 anytime curve 上，HBWS 达到预注册部署质量阈值所需的中位搜索美元成本低于 full-fidelity evolution 至少 40%，且最终 AUBPC 非劣（δ=3pt）。阈值在 Direct/人工 baseline 冻结后、运行搜索前写入 prereg。

**H3b（Protocol B）**：相同总搜索 cap 下，HBWS 的冻结 policy 在两预算 AUBPC 上优于 Static Evolution Search 的预算拆分方案。

**H3c（摊销）**：在质量满足非劣的前提下，若每任务部署成本节省 Δc>0，则

\[
N^*=\left\lceil\frac{C_{search}^{HBWS}-C_{design}^{baseline}}{\Delta c}\right\rceil.
\]

若 Δc≤0 或质量不满足非劣，(N^*=\infty)，不得展示虚假的正摊销。对预算分布 (p(b)) 计算加权 Δc，并给出 10/50/90% 三种部署混合情景。

### 6.4 统计单位与多重比较

- 主单位是任务；同一任务上的方法、预算和执行 seed 尽量成对。
- 最终成功率 CI 使用按任务分层的 paired bootstrap；数学按学科、代码按难度/来源分层，10,000 次重采样。
- 执行随机性：每个冻结 policy 至少 3 个执行 seed；先对同任务执行 seed 平均，再按任务 bootstrap。
- 搜索随机性：HBWS、Static、Random 各 3 个独立搜索 seed。主表展示每个搜索 seed 输出 policy 的结果及均值；层次 bootstrap 以搜索 seed→任务→执行 seed 重采样作为敏感性分析。仅 3 个搜索 seed 时，不夸大对任意重跑的总体保证。
- 确认性比较族：H1、H2、H3 分族使用 Holm 校正；OOD 和机制案例除非 prereg 明确，否则标为次要/探索性。
- 同时报成功率差、风险比或 odds ratio、每成功任务成本差，不能只报 p 值。
- 搜索期间只用 search-dev；top-K 候选按冻结规则在 validation 选择一次；最终 ID/OOD test 只运行冻结候选。

### 6.5 人工设计负担的可测量代理

若要声称“减少人工设计负担”，必须记录：人工种子数、每个种子的编写分钟数、搜索后人工修改次数、逐预算人工调参次数、人工 Workflow 总数。主协议固定每任务族最多 3 个种子，method-freeze 后禁止按预算人工编辑。工程基础设施开发时间单独记录，不折算成搜索美元，也不与算法搜索成本混为一谈。

---

## 7. Protocol A / B

### 7.1 公共控制条件

所有方法共享：执行模型的精确版本/SKU、system prompt 外壳、可用工具、公开反馈、sandbox、上下文截断、预算账本、任务顺序、数据切分、最终评估器和 API 重试规则。搜索方法共享声明式 DSL、合法变异集合和初始化种子；static 版本仅禁止读取预算状态的条件边。

搜索期间候选结果可跨 fidelity 复用；是否使用 prefix cache 必须对所有搜索方法一致，并同时报告“逻辑未缓存成本”和“实际账单成本”。最终部署测试默认关闭跨任务答案缓存。

### 7.2 Protocol A：部署能力上界

目的：问一个共享 budget-contingent policy 能否接近或超过为每个预算专门设计的静态上界。

- HBWS 每个搜索 seed 获得搜索 cap (S)，联合 `$0.10/$0.25` 搜索一个 policy。
- Static Evolution Search 在每个预算档分别获得完整 cap (S)，即每 seed 总计 (2S)。
- static 可独立选择每档不同拓扑和 Prompt；这是有意偏向 static 的上界协议。
- 比较冻结 policy 在 `$0.10/$0.25` 的成功、实际成本、P50/P95 时延、预算违约、AUBPC、(R_{budget}) 和 (R_{max})。
- Protocol A 不能用来声称 HBWS 搜索更省钱，因为搜索预算不匹配。

### 7.3 Protocol B：设计效率

目的：问需要同时覆盖多个部署预算时，谁在相同总设计预算内找到更强的可部署方案。

- HBWS 每 seed 总搜索 cap 为 (S)。
- Static Evolution Search 每 seed 总 cap 也是 (S)，预注册平均拆成 (S/2+S/2)；不得看结果后重新分配。
- Random Search 同样获得 (S)，候选仍在两个预算联合评估。
- Static Protocol B 的结果可取自 Protocol A 运行到每预算 (S/2) 的 checkpoint，避免重复账单；checkpoint 必须在看到后续结果前自动保存。
- 比较 search-cost vs AUBPC anytime curve、达到质量阈值的成本、最终部署质量和摊销临界点。

Protocol A 与 B 必须分别成表，不能挑选其中更有利的一套作为唯一主结果。

---

## 8. 实验设计

### 8.1 数据与切分

**代码族：** MBPP+ 为 ID 搜索/验证/测试，HumanEval+ 全集为 OOD；以 pass@1 和扩展测试判分。目标切分为 MBPP+ 120 search-dev / 40 validation / 至少 150 final-test，剩余样本保留不用。若官方当前版本可用题数变化，按固定比例和 hash 重新生成一次并冻结。

**数学族：** MATH 难度 4–5（优先 5）中的四个预注册学科用于 ID，目标 120/40/150；另两个完整未见学科抽取 100 题作 OOD。严格按 problem ID 去重，检查近重复模板；gold 只在外部评估器。

search-dev 内按难度和类别生成 24→64→120 的嵌套子集。切分脚本、随机 seed、原始数据版本、license 和 SHA256 写入匿名复现包。搜索任何一次启动后不得重切分。

### 8.2 方法矩阵

| 方法 | 自动搜 Prompt | 自动搜拓扑 | 预算条件控制流 | 跨预算联合目标 | 3 搜索 seed | 角色 |
|---|---:|---:|---:|---:|---:|---|
| Direct Prompt | 否 | 否 | 否 | 否 | n/a | 最低基线 |
| CoT + self-check | 否 | 否 | 可预算安全停止 | 否 | n/a | 强单 Agent |
| Human Strong Workflow | 否 | 否 | 否 | 否 | n/a | 人工 Harness |
| Prompt-only Search | 是 | 否 | 否 | 按协议 | 建议 3 | 搜索维度消融 |
| Topology-only Search | 否 | 是 | 按版本 | 按协议 | 建议 3 | 搜索维度消融 |
| Random Search | 是 | 是 | 是 | 是 | 必须 3 | 搜索器基线 |
| Static Evolution Search | 是 | 是 | 否 | 否/预算拆分 | 必须 3 | 核心受控基线 |
| HBWS | 是 | 是 | 是 | 是 | 必须 3 | 主方法 |

AgentEvo 若存在可公平复用的 artifact，作为外部兼容性实验；否则不冒充完整复现，也不影响主因果比较。

### 8.3 必做消融

1. **No-CB-objective**：DSL 仍允许预算分支，但每档独立/平均 score 选择，移除最差预算项和 cross-budget credit。
2. **No-budget-state**：保留相同拓扑搜索，但 policy 不读取预算比例和 reserve mask。
3. **Prompt-only**：固定人工强拓扑。
4. **Topology-only**：固定 Prompt 注册表。
5. **No-multifidelity**：从第一轮使用完整 search-dev；以 anytime cost-to-quality 比较。
6. **Mean-only**：去掉 LCB 和不确定性 racing，用均值选择。

若成本吃紧，1、2、5 是不可删除的核心消融；其余可降为单任务族，但必须在 prereg 前决定。

### 8.4 预算 profiles

下列是 pilot 前工程假设，不是事实。单价、模型上下文限制和 Azure 部署行为必须在第 1 周用官方价格页与实际账单核验。

| Profile | USD | LLM calls | input tokens | output tokens | tool calls | wall time |
|---|---:|---:|---:|---:|---:|---:|
| tight / search | 0.10 | 4 | 8,000 | 2,000 | 4 | 90 s |
| unseen / test only | 0.15 | 6 | 12,000 | 3,000 | 5 | 135 s |
| loose / search | 0.25 | 8 | 16,000 | 4,000 | 6 | 180 s |

如果冻结模型的最坏单调用无法在 tight profile 中完成 `reserve`，应降低节点输出上限或更换预算 profile；不能允许先调用再超帽。`$0.15` 的所有非美元维度由 tight/loose 线性插值并在 prereg 中固定。

### 8.5 全局费用预算（pilot 前工程估算）

全局 API/工具账单硬帽暂定 `$2,000`；80% 预警、90% 禁止启动可选项、100% 熔断。此表是资金分配，不是实验结果：

| 项目 | 暂定额度 | 说明 |
|---|---:|---|
| 节点成本 pilot 与基线校准 | $80 | 两任务族、各节点抽样 |
| HBWS 主搜索，3 seeds | $300 | 每 seed cap (S=$100) |
| Static Protocol A，3×2 runs | $600 | 每预算每 seed `$100`；其中 `$50+$50` checkpoint 构成 Protocol B |
| Random Search，3 seeds | $300 | 与 HBWS 相同，每 seed cap (S=$100) |
| 核心消融 | $150 | 优先 No-CB、No-budget-state、No-multifidelity |
| 最终 ID、`$0.15`、OOD 与重跑 | $400 | 按实际成本；任务 cap 不是平均花费 |
| 统计/写作期间必要复算 | $50 | 仅冻结候选 |
| 不可预见与失败重试储备 | $120 | 不用于扩展实验 |
| **合计** | **$2,000** | pilot 后冻结 |

成本模型输出三列：按任务 cap 推导的绝对上界、pilot P95 预测、pilot 均值预测。正式规模只能根据 P95 列冻结。人工工时另报，不货币化混入 API search cost。

### 8.6 必报指标和案例

- 成功率、分层 paired bootstrap CI、效应量；
- input/output token、美元、LLM/工具调用、P50/P95 时延；
- policy-attributable 与 provider-attributable 预算违约率；
- AUBPC、cross-budget empirical archive、budget-regret；
- 全部搜索成本、optimizer 成本、失败候选成本、anytime curve；
- 摊销临界点及部署预算混合敏感性；
- 分支触发率、循环次数、fallback/END 原因、不同预算的结构路径；
- 至少 3 个成功案例、3 个失败案例。

案例不能手挑。预先定义失败类型：错误验证信号、预算过早耗尽、reserve 过保守、循环不收敛、投票同质错误、aggregation 覆盖正确 incumbent、OOD 路由失效、API/沙箱外部失败。每类按全测试中出现频率排序，成功案例从“低预算提前停”“高预算追加验证修正”“未见预算平滑回退”三种机制中各取一个最接近簇中心的代表；失败案例取最高频三类的代表。公开完整 trace 的脱敏版本。

---

## 9. 6 周甘特图与 Go/No-Go

ICLR 2027 官方投稿日期截至 2026-08-04 尚未在官方页面核验到；`2026-09-15` 仅作为内部冻结日，不能写成官方截止日期。每周范围按当前日期倒排。

| 周 | 必须完成 | 交付物 | Go/No-Go |
|---|---|---|---|
| W1：8/4–8/10 | prior-art 精读；审计现有代码/“已实现”声明；冻结 DSL、账本、数据、价格、prereg；节点成本 pilot | `PREREGISTRATION.md`、数据 hash、价表、预算安全测试、成本 P95 表 | **G0**：AgentEvo/BAVT/Agent-UCT 差异仍成立；**G1**：10k fuzz/模拟执行零 policy 预算违约。否则先修协议，不启动搜索 |
| W2：8/11–8/17 | Direct、CoT、Human、Prompt-only、Topology-only；HBWS 单族小试；跨预算 credit 单元测试 | baseline 表、HBWS smoke archive、自动日报 | **G2**：HBWS 在至少一族的 F2 上改善 AUBPC 或 budget-regret；否则只允许一次目标/算子修订并更新 prereg 偏差记录 |
| W3：8/18–8/24 | Protocol B 的 HBWS/Static/Random 三搜索 seed；核心消融启动 | equal-search-budget anytime curves | **G3**：No-CB 或 No-budget-state 至少一个出现预期方向，且预算分支非零触发。否则 best-paper 叙事降级 |
| W4：8/25–8/31 | Static Protocol A 继续到每预算完整 cap；完成两族主搜索和消融；冻结 top-K | Protocol A/B 全部 dev/validation 结果、`method-freeze` tag | **G4**：H2a 在 validation 有希望且至少 H1/H3 一项过门；否则转严谨负结果/benchmark 论文。8/31 后不改方法 |
| W5：9/1–9/7 | 一次性运行 ID test、`$0.15`、OOD；统计、3+3 案例、复现冷启动 | 主表、Fig.2–5、artifact dry run | **G5**：无泄漏、无关键账本 bug、主结果可从原始日志重算。失败只允许修评估 bug并全量重跑受影响方法 |
| W6：9/8–9/15 | 论文、附录、匿名仓库、claim-evidence audit、外部复现 | 投稿 PDF、匿名复现包、rebuttal 风险清单 | **G6**：每项主张有对应表/图/检验；否则删 claim，不补看后实验 |

**必须完成：** DSL、联合目标/credit、Protocol A/B、两个任务族、`$0.15`、3 搜索 seed、核心基线/消融、账本、主图和匿名复现包。

**时间允许：** SWE-bench Lite、第二模型、WebArena、LLM 变异、MCTS/RL、额外理论。只有 G4 通过且主线预算/时间有明确余量才启动；优先级依次为第二模型小规模验证、SWE-bench Lite、其他。

---

## 10. 风险与降级路径

| 风险 | 早期信号 | 处置 | 允许的论文定位 |
|---|---|---|---|
| AgentEvo 已覆盖大部分主张 | 精读发现其 policy 已显式以运行时预算条件化并跨预算联合搜索 | 缩小 claim 到硬预算、Protocol A/B、budget-regret；若仍无实质差异则停止“新方法”叙事 | 评估协议/负结果论文 |
| budget 分支很少触发 | 两预算路径基本相同 | 检查 profile 是否真正区分；不得人为制造结果 | 证明静态 policy 已足够，作为边界发现 |
| HBWS 只在一档好 | cross-budget Δ 一正一负 | 保留结果；No-CB 对照定位目标失败原因 | 不声称 one-policy-many-budgets 成功 |
| 数学 heuristic critic 误导 | refine 后正确率下降 | 强制保留 incumbent；报告 critic calibration | 代码族主结果，数学为负结果 |
| reserve 过保守 | 大量本可完成调用被拒 | 用可证明的更紧上界，不用事后超帽 | 安全—利用率权衡 |
| API 模型漂移/价格变化 | 同 prompt 输出分布或账单变化 | 固定 deployment 版本；每日记录 model/version；价格双报 token 与冻结美元 | 限定版本结论 |
| 搜索过拟合 search-dev | validation/test 大幅下降 | 不调 test；报告 selection bias 与 archive 稳定性 | 搜索泛化分析 |
| 3 search seeds 方差极大 | 不同 policy 差异超过任务 CI | 降低对搜索算法稳定性的主张，展示全部 seed | 部署候选而非普遍算法主张 |
| `$2,000` 不够 | pilot P95 超预算 | 按 prereg 顺序缩减 optional → 次要消融 → ID test 150→100；不删主对照/3 seeds/`$0.15` | 规模较小但完整协议 |
| 六周内没有正结果 | G4 不过 | 冻结方法，转“hard-budget workflow search benchmark + negative findings” | TMLR/AAMAS/ICLR empirical track 风格 |

任何降级都保留全部失败 seed 和日志，不进行隐性重启或只报告幸运 policy。

---

## 11. 论文结构与关键图表

### 11.1 正文目录

1. Introduction：人工 Harness → 人定义安全空间、系统搜索架构；三个贡献。
2. Related Work：automated workflow/agent design；cost-aware evolution；budget-conditioned inference；search efficiency。
3. Problem Formulation：Budget-Contingent Workflow Policy、硬预算、AUBPC、budget-regret。
4. HBWS：DSL、预留账本、三阶段进化、cross-budget credit、多保真 racing。
5. Experimental Protocol：Protocol A/B、任务、基线、统计、成本定义。
6. Results：部署能力、设计效率、未见预算、OOD、消融。
7. Mechanistic Analysis：路径触发、案例、失败类型、reserve 利用率。
8. Limitations and Broader Impact：有限 DSL、两任务族、API 漂移、自评可靠性、搜索碳/成本。
9. Conclusion。

附录：完整 DSL schema、Prompt、价表、预算安全证明、所有 seed、统计细节、额外 archive、匿名复现说明。

### 11.2 图表及其证明责任

| 图/表 | 内容 | 必须支持的结论 |
|---|---|---|
| Fig.1 | 一 policy 在三个预算下的不同安全路径 + 搜索总览 | 区分 budget-conditioned policy 与 per-budget static graph |
| Fig.2 主图 | Protocol A：HBWS 单 policy vs 两个 per-budget static upper candidates 的预算—成功曲线；含 `$0.15` | 部署能力、非劣/优越、未见预算 |
| Fig.3 主图 | Protocol B：search dollars/逻辑执行数 vs AUBPC anytime curves | 联合搜索的设计效率，不混用部署成本 |
| Fig.4 | cross-budget credit 消融、路径触发 Sankey/频率、reserve 拒绝原因 | 改善来自跨预算机制，而非只多花调用 |
| Fig.5 | 摊销曲线，按部署预算混合给 (N^*) 和不成立区域 | 搜索投入何时值得；无节省时明确无 break-even |
| Fig.6 | ID/OOD/未见预算森林图 | 泛化范围和不确定性 |
| Tab.1 | 经核验的 Related Work 差异矩阵 | 诚实定位，尤其 AgentEvo/BAVT/Agent-UCT |
| Tab.2 | Protocol A 主结果：成功、CI、cost、tokens、calls、P50/P95、violations、regret | 公平运行时比较 |
| Tab.3 | Protocol B：search cost、候选数、invalid rate、AUBPC、cost-to-threshold | 公平设计效率比较 |
| Tab.4 | 消融和三个 search seed 全量结果 | 组件必要性与稳定性 |
| Tab.5 | 3 成功 + 3 失败案例索引和失败类型频率 | 机制与边界，不做轶事挑选 |

---

## 12. 匿名复现包验收清单

- 声明式 DSL schema、validator、fuzzer、预算安全测试；
- 账本、reserve/settle、并发预留和 watchdog；
- 固定数据清单、版本、license、hash 和切分脚本；
- 全部 Prompt registry、人工种子、变异 patch 与 lineage；
- Protocol A/B 配置及搜索 cap；
- 三个搜索 seed 和三个执行 seed 的原始事件日志；
- 从日志重算所有表图的一条命令；
- 冷启动 smoke test，不需要作者私有路径；
- `.env.example` 只含变量名，不含 endpoint/key；
- secret scan、依赖 lockfile、模型版本和价表快照；
- DATA_CARD、MODEL_CARD、LIMITATIONS、失败运行清单；
- 一位未参与实现者按 README 复现至少一个小规模表格。

---

## 13. 英文 Abstract

### 13.1 当前写作版（无结果数字）

Complex agentic workflows are still largely hand-engineered: developers choose the number and roles of agents, prompts, graph topology, verification steps, loops, and stopping rules, often redesigning the workflow when deployment constraints change. We study a narrower alternative to fully autonomous agent design: humans specify a safe and auditable design space, while a meta-agent searches for an executable architecture within it. We formulate **budget-contingent agent architecture search**, where the search target is a single workflow policy that conditions its control flow on execution state, admissible feedback, and remaining per-instance resources. We introduce HBWS, a constrained evolutionary method that jointly evaluates every candidate across multiple deployment budgets, assigns mutation credit from conservative cross-budget improvements, and uses multi-fidelity racing to reduce search cost. A reservation-based ledger enforces token, call, tool, latency, and monetary caps before node execution. We separate two questions that are often conflated: deployment capability, where static workflows may be optimized independently for each budget, and design efficiency, where all methods receive the same total search budget. Across code and mathematical reasoning tasks, we evaluate deployment quality, unseen-budget generalization, search cost, and amortization against hand-designed, random, prompt-only, topology-only, and matched static-evolution baselines. Our study provides a controlled test of when one searched policy can replace repeated per-budget workflow design—and when it cannot.

### 13.2 投稿版（结果占位符；不得在有结果前填）

Complex agentic workflows are still largely hand-engineered, and their topology, prompts, verification logic, and stopping rules are commonly redesigned as deployment constraints change. We formulate **budget-contingent agent architecture search**, where the target is a single safe and auditable workflow policy that conditions its control flow on execution state, admissible feedback, and remaining per-instance resources. We introduce HBWS, a constrained evolutionary method that jointly evaluates candidates across budgets, assigns mutation credit from conservative cross-budget improvements, and uses multi-fidelity racing under a reservation-based hard-budget ledger. We evaluate two distinct settings: a deployment-capability protocol that grants static workflows independent search at each budget, and a design-efficiency protocol that matches total search cost. On code and mathematical reasoning benchmarks, a single HBWS policy is non-inferior to per-budget static search at both preregistered budgets within a [δ]-point margin and improves success by [X] points at [BUDGET], while retaining [Y] performance at the unseen intermediate budget. Under matched search cost, HBWS improves cross-budget AUBPC by [Z] and reduces the cost to reach the deployment-quality threshold by [R]%. The resulting deployment savings amortize the additional search cost after [N] tasks under the preregistered budget mixture. Ablations attribute the gains to cross-budget credit assignment and budget-conditioned control flow rather than additional inference calls. These results delineate when a single searched workflow policy can replace repeated per-budget workflow design under hard operational constraints.

若任何占位结论未过相应检验，必须删除整句，而不是换成模糊的 “competitive”。

---

## 14. 已核验事实、待核验事实与来源

### 14.1 截至 2026-08-04 已核验

- AFlow 是 ICLR 2025 论文，公开描述为以 MCTS 和代码修改搜索 Workflow：<https://proceedings.iclr.cc/paper_files/paper/2025/hash/5492ecbce4439401798dcd2c90be94cd-Abstract-Conference.html>
- GPTSwarm 将语言 Agent 表示为图并优化节点 Prompt 与边：<https://proceedings.mlr.press/v235/zhuge24a.html>
- AgentSquare 提出模块化 Agent 搜索、模块进化和重组：<https://arxiv.org/abs/2410.06153>
- EvoFlow 使用 niching evolutionary algorithm 搜索异构、复杂度自适应 Workflow 群体：<https://arxiv.org/abs/2502.07373>
- AgentEvo 于 2026-05-22 发表，公开描述包括成本感知多阶段进化、性能—token cost archive，以及 MATH/HumanEval/MBPP 等实验：<https://link.springer.com/article/10.1007/s40747-026-02325-0>
- BAVT 以剩余资源比例条件化推理树节点选择：<https://arxiv.org/abs/2603.12634>
- Agent-UCT 使用成本感知 UCT、配置前缀复用和缓存优化：<https://arxiv.org/abs/2607.24162>
- ICLR 官方 Future Meetings 页面目前只给出 ICLR 2027 地区，未在本次检索中给出投稿截止日：<https://iclr.cc/Conferences/FutureMeetings>

### 14.2 必须在 Week 1 再核验并存档

- ADAS、EvoFlow、AgentEvo、BAVT、Agent-UCT 的最新论文版本、正文细节和 artifact 状态；
- ICLR 2027 官方 CFP、abstract/paper 截止日期、页数、匿名和生成式 AI 政策；
- Azure 具体 region、deployment SKU、缓存输入、batch、工具费和税前价格；官方价格页：<https://azure.microsoft.com/en-us/pricing/details/azure-openai/>；
- MBPP+/HumanEval+/MATH 的当前版本、可用题数、许可和污染风险；
- “GPT-4o”是否仍是实际固定 deployment；论文必须写精确 snapshot/SKU，不能只写产品族名。

---

## 15. 最终执行原则

1. 先冻结 claim 和检验，再看主结果。
2. 同一个 policy 的定义必须由代码保证，不能在不同预算加载不同 Prompt/图文件。
3. `$0.15` 对搜索器、阈值选择和候选选择完全不可见。
4. Protocol A/B 同时报告，不用预算不匹配的胜利替代公平比较。
5. AgentEvo 不忽略、不稻草人化；无法公平复现时诚实说明。
6. 任何预算违约、失败 seed、无效候选和搜索重启都进入日志。
7. 六周内优先做完主线，不以 SWE-bench、WebArena、第二模型或复杂搜索器掩盖核心假设未成立。
