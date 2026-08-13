# 交接文档 — "Repair, Breakage, and Reference Preservation" (ICLR 2027)

> **这份文档用作新会话的起始 prompt。** 读完后你应当知道：项目在哪、论文现在主张什么、
> 已经踩过哪些坑、什么已验证什么还开放、以及哪些事绝对不能做。
> 本文档写于 2026-08-13，全面替换了 2026-08-06 的旧版（旧版数字、标题、成本口径均已过时，
> 不要参考仓库里任何早于本次提交的 HANDOFF 副本）。

---

## 0. 硬性约束（最优先，违反会造成真实损失）

1. **绝对不要在未获用户明确许可前调用任何付费 API。** 每次新的真实花费都要先给出预估
   金额，获得用户逐次同意才能执行。
2. **用户的个人 Claude/Anthropic key 绝不能被这个项目使用。** `API_Key.txt`/`.env` 里
   如果混有 Anthropic key，只读不用。
3. **仓库已连接 GitHub remote**：`https://github.com/yangcao2311/Budget-Aware-Agent-Architecture-Search.git`
   （`origin`，**Private**，ICLR 双盲要求必须保持私有）。git 身份是仓库级设置
   （`user.name=agent`，`user.email=agent@localhost`），不要改成真实姓名。**push 前必须
   先告知用户在做什么并等待确认**——push 是对外部共享状态的操作。
4. **论文正文（`paper/main.tex`）里绝不能出现任何第三方 API 中转/代理商名称**
   （chatanywhere / gptproto / anyrouter / tokenrouter 等），也不能提具体的 free-tier
   /速率限制细节。Model 段落只能写 "All experiments use GPT-4o." / 附录里 Kimi 部分只能
   写 "Kimi K3"，不要写任何渠道信息。
5. **不要重新生成数据切分。** `hbws/data.py` 会拒绝覆盖已存在的切分，这是有意为之。
6. **不要修改已冻结的预注册文档正文**（`PREREGISTRATION.md` §1–§8）。只能在 §9 偏差
   日志追加（append-only），且每次追加要写清楚触发原因、协议内容、时间戳。
7. **绝不能把"筛选后好看的结果"当成完整结果集呈现而不加说明。** 之前有一次用户自己
   提出想筛选 <5% 错误率的条件，被指出诚信风险后当场收回，改为要求把全部条件补跑到
   接近 100% 完成度（见 §3 Kimi 补跑）。这条判断标准要在类似请求出现时主动提醒用户。
8. **Kimi/tokenrouter 渠道并发必须锁定为 2**（`--workers 2`），这是用户明确要求的速率
   限制。固定退避重试参数：`LLM_BACKOFF_SCHEDULE=10,30,90 LLM_MAX_RETRIES=3`。
9. **任何 API key 只要在会话文本/工具输出里完整出现过，就视为已泄露，必须提醒用户
   轮换。** 本次会话中 tokenrouter 的 key 完整出现在工具输出里
   （`sk-93HUUUBh...`，见 `.env` 的 `KIMI_API_KEY`），**如果还没有轮换，新会话开始时
   应主动提醒用户轮换**。

---

## 1. 项目位置与结构

工作目录 `/home/ycao95/Agent`（本地 git 仓库，remote 见上）。Python 环境 `.venv`。

```
hbws/            库：DSL 与校验器、预留式账本、图执行器、验证器、评测协议、搜索
scripts/         每个实验/分析一个入口（见 §6），含 audit_claims.py 全量复算脚本
data/            冻结切分 + SHA256SUMS（一次性生成，永不重生成）
experiments/     全部原始逐题结果、预算账本、搜索登记表、Kimi 补跑数据
paper/           main.tex、refs.bib、图、main.pdf（20 页，正文 9 页）
PREREGISTRATION.md   预注册 + 只增不改的偏差日志（§9 记录了三臂因果实验协议、
                     Kimi 补跑协议、best-of-3 措辞修正等全部投稿后偏差）
HANDOFF.md           本文件
```

**仓库已精简**：`backup_0806/`、旧版 `AMENDMENT_01.md`/`HANDOFF.md`/`执行方案.md`/
`研究说明.md`/中文论文版、两份无关第三方论文 PDF、过时的图版本、构建产物等，均已从
git 追踪移除（本地磁盘上部分文件仍保留，只是不再进 GitHub）。**`paper/` 整个目录也
已从 GitHub 移除**（用户明确要求：代码仓库只放代码+数据，论文稿件不放进去）——
`main.tex`/图/`main.pdf`/ICLR 模板只存在于本地磁盘，不在远端。**GitHub 上现在只有**：
`hbws/`、`scripts/`、`experiments/`、`data/`、`PREREGISTRATION.md`、`README.md`、
`requirements.txt`。**新会话如果要改论文，改的是本地 `paper/main.tex`，改完不要假设
它会自动同步到 GitHub——需要显式加回来 push，且每次都要先确认用户是否真的想把论文
放回代码仓库。**

---

## 2. 论文现在主张什么

**标题**：*Repair, Breakage, and Reference Preservation in Verify–Refine Workflows*
（已从旧标题 "Reference-Preserving Verification Under Hard Inference Budgets" 改名，
是对一轮模拟外部评审的回应之一）。

**核心分解**（对二元判分是精确恒等式，不是拟合）：

```
Δ = (1−p)·r − p·b        r=修复率, b=破坏率, p=基线准确率
```

**再拆一层**（机制核心）：

```
b = Pr(接受, I错 | B对) + Pr(拒绝, J错 | B对)
```

第一项在 **I = B**（初稿就是参考系统的实际输出）时恒为零，此时

```
b ≤ Pr(误拒 | 基线正确)
```

称为**参考保持（reference preservation）**——通过*出身*（保留参考系统真实输出，只在
验证器拒绝后才修改）而非某个控制流特征来约束破坏率。这是一个**包含关系论证**，不是
深层定理——已在论文里明确降级措辞，不再称"定理"。

**三臂因果溯源实验**（本次会话新增的最强证据，真实数据）：显式复用存储输出 / 同策略
重新生成 / 不同策略生成，三臂下游结构完全一致，只有 incumbent 的来源不同：

| | 代码破坏率 | 数学破坏率 |
|---|---|---|
| 显式复用（arm 1） | 0.000 | 0.003 |
| 同策略再生成（arm 2） | 0.028 | 0.027 |
| 不同策略再生成（arm 3） | 0.109 | 0.017 |

代码族清晰单调；数学族三臂都接近零，与"非 oracle 自检验证器的误拒界本就很紧"一致，
论文里已如实解释这个非单调现象，没有掩盖。

**贡献顺序**（回应评审后重排）：
1. 发现"光有闸门不够"——验证器保护的是**谁的答案**才决定是否可能倒退
2. 把这个发现写成精确的记账恒等式
3. 实证刻画（confirmatory C1–C5、three-sample 事后对照、跨模型复现）

论文**明确划清范围**：不解决"同成本下哪种推理策略最好"，只解决一个正交的必要问题——
"怎么安全地评估/采纳一个 verify-refine 工作流"。§sec:threesample 报告了一个零新增
成本的事后采样对照，但**明确声明这不是同成本对照**。

---

## 3. 本次会话完成的主要工作

### 3.1 三臂因果溯源实验（真实数据，约 $18.6 真实花费）

`scripts/run_provenance_causal.py` + `scripts/provenance_causal_analysis.py`
（另有 `provenance_arm_c_leak.py`、`provenance_from_logs.py`、`regen_leak.py` 做低/
零成本复分析）。新增 DSL 节点 `assign`（`hbws/dsl.py` `wf_assign_refine`，
`hbws/runner.py` 里零成本注入 `task["_assign_solution"]`）。结果已写入
`§sec:theory` 的 "The three-arm causal test" 段落。

### 3.2 回应一轮详细的模拟外部评审（5/10，临界不过）

改动：标题、贡献顺序、压缩自动搜索段落、新增
`§sec:threesample`（事后 best-of-3 对照，明确标注非预注册、非同成本，两个方法分别
命名而非笼统称"best-of-3"，报告了任务配对 bootstrap 置信区间）、Limitations 与
Conclusion 相应更新。`scripts/best_of_3_zero_cost.py` 是这部分的分析脚本。

### 3.3 第二模型家族（Kimi K3）全量补跑

- 触发原因：用户曾提议筛选 <5% 错误率的条件展示，被指出诚信风险后主动要求补跑到
  接近完整（**不要重复提议筛选展示**）。
- 补跑协议冻结在 `PREREGISTRATION.md` §9（2026-08-13 条目）：只重试之前 provider
  报错的 task×seed×arm，其余全部不动，并发 2，固定退避 10/30/90s、最多 3 次重试，
  append-only 尝试日志 `experiments/kimi_backfill_attempts.jsonl`。
- 驱动脚本 `scripts/kimi_retry_failed.py`，最终 10/10 条件完成率 97–100%。
- 结果已写入附录 `app:kimi`，措辞是"复现了跨验证器敏感性的定性模式，不是因果实验
  （未在第二模型上重跑），也不是量级（GPT-4o 内部的迁移测试已证明量级本就不可迁移）"。

### 3.4 仓库精简 + 首次公开（Private）推送

见 §1。已推送两次：`5b51fa9`（首次批量上传）→ `9fda52d`（精简 + Kimi 数据合并，
本次会话完成）。

---

## 4. 验证状态

```
.venv/bin/python scripts/audit_claims.py
# 252/252 claims verified against raw logs
```

论文正文严格 9 页（ICLR 上限），全文（含附录+参考文献）20 页，已用
`pdftotext -f 9/10 -l 9/10 -layout` 反复核实第 9 页结尾是 Conclusion、第 10 页开头
是 References。**任何编辑 main.tex 后都要重跑 audit_claims.py 和这个页数检查。**

页数预算极度紧张、且页面断行行为高度非线性（缩短文字有时对渲染行数完全无影响，
似乎被别处的重新对齐吸收）——改完一次要重新完整编译（删 `.aux/.log/.out/.bbl/.blg`
后 `pdflatex`×2 + `bibtex`）确认不是缓存假象。

---

## 5. 成本口径

真实账单约 $300+（因果实验后又新增约 $18.6 真实 / 约 $26.6 逻辑成本用于三臂实验；
Kimi 补跑成本另计，量级远小于 GPT-4o 主实验）。**论文表格里的每题部署成本用逻辑成本
（假设无缓存该花多少）；"这项研究总共花了多少"用真实账单。** 两者不要混用。

---

## 6. 常用命令（全部无需 API）

```bash
.venv/bin/python scripts/audit_claims.py        # 252 个数字 vs 原始日志
.venv/bin/python scripts/confirm_partI.py       # C1–C5 判定
.venv/bin/python scripts/confirm_partII.py      # P1–P5 判定
.venv/bin/python scripts/false_rejection.py     # 界 b ≤ Pr(误拒) 检验
.venv/bin/python scripts/best_of_3_zero_cost.py # 事后 best-of-3 对照 + 配对 CI
.venv/bin/python scripts/provenance_causal_analysis.py  # 三臂因果实验复分析
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 7. 待办（截至本次交接）

- **A2 逐句 claim-evidence 复核**：`audit_claims.py` 只查数字，不查因果措辞强度，
  建议投稿前人工扫一遍所有 "because / therefore / shows that"。
- **A3 匿名复现包冷启动测试**：`README.md` 里"一条命令重算全部"的流程从未在干净
  环境跑过。
- **A4 相关工作二次核实**：几篇 2026 年引用只经检索确认存在，未读全文，投稿前需
  核实。
- **是否需要把因果实验/Kimi 补跑的完整过程也同步进 `README.md`**（当前 README 可能
  还停留在旧版实验清单，需要核对）。
- 用户即将更换服务器，本地 `Agent/` 目录会被清空——**GitHub 私有仓库是权威副本**，
  新机器上应 `git clone` 而不是假定本地文件还在。

---

## 8. 与用户协作的注意事项

- 用户不是这个子领域的专家，解释实验要拆解到"为什么选这个数据集""为什么要三个
  种子"这一层级。
- 用户会转发外部评审意见让你判断对错，**先核实再回答**，不要无条件接受评审方的
  表述。
- 报告数字时务必区分逻辑成本与真实账单。
- 用户明确要求：**被问判断时直接给判断，不要先给一堆方案。**
- 涉及花钱、push、删除、覆盖已有数据的操作，**一律先说明在做什么、预估影响，
  等用户确认**，不要自作主张执行。
- 涉密key只要在对话里完整出现过就视为泄露，主动提醒轮换，不要重复使用。
