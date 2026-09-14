# 贡献约定

给接手这个仓库的人（或 AI 编码助手）看。改动前先读完这一页。

## 一条不能破的铁律

**大模型永远不做算术、不估数、不外推。**

所有数值只能来自 `src/api/services.py`（L2 算法内核层）返回的 JSON。
`src/agent/` 里的任何代码都不许自己计算数字，也不许把模型生成的文本当数据用。

违反这条会同时破坏三件事：幻觉率、可解释性、可审计性。
`src/agent/guard.py` 是这条铁律的执行者，`tests/test_agent.py::TestGuard` 是它的守门人。

推论：**回答里出现的每一个数字都必须有出处**。
如果模板要写"近 14 天日产"，那么 14 也得是工具返回的字段（`rate_window_days`），
不能是代码里的字面量 —— 否则 guard 会正确地把它报成幻觉，而那不是误报。

## 分层与依赖方向

```
src/agent/  →  src/api/services.py  →  src/models,reserves,sec  →  src/db.py
```

依赖只能从左往右。特别地：

- `src/models`、`src/reserves`、`src/sec` **不许 import `src/agent`**，也不许知道大模型的存在。
- SEC 单元"新-老-措"构成评估的纯计算在 `src/reserves/workload.py`（工作量剥离）、
  `src/sec/composition.py`（构成、对账、敏感性、归因）、`src/sec/indicators.py`（指标评分）；
  它们不读库，由 `services.py` 准备好月度表、井表、事件表后传入。
- **写操作只有一个**：`services.persist_unit_evaluation`（评估结果入库），只给 CLI `unit-eval` 用，
  不注册为智能体工具、不开放 HTTP 接口。智能体工具一律只读。
- `src/api/services.py` 是唯一的数值出口。新增能力时先在这里加函数，再在
  `src/agent/tools.py` 注册成工具，最后在 `src/agent/plans.py` 挂进计划模板。
- 任何 service 返回体都必须带四个追溯字段
  （`model_version` / `label_def_version` / `data_source` / `trace_id`），
  由 `_env()` 统一注入。`tests/test_agent.py::TestServiceEnvelope` 会检查。

## 口径

**分位数口径全仓库统一**：`p10 / p50 / p90` 就是分位数本身，`p10` 是数值小的那个。
储量行业习惯的"P90 = 低估计"用 `low_estimate` / `high_estimate` 别名表达。
两套口径混用是储量工作里最常见的低级错误，见 `src/reserves/dca.CONVENTION`。

**标签口径在 `conf/label_def.yaml`，不在代码里。**
改口径必须同时改 `version`，历史标签按版本并存。
永远不要为了让某个指标好看去改标签定义而不升版本号。
当前为 v2：EUR 是自然递减口径（有措施的井只用首次措施前的数据），措施增油由单元构成评估单列。

**SEC 单元口径：**

- 评估对象层级（公司 → 采油厂 → SEC 单元）只读 `sec_unit` / `unit_well` 两张表；`conf/units.yaml` 只给合成适配器用。
- 已证实储量一律取低值：老井逐井"指数递减"与"最佳估计"取低值；新井"动态法或类比法"与"模型法低估计"取低值；
  老井基础里本期做过措施的井用**措施前**基线，措施带来的部分只在措施增储里算一次。
- 对账里技术修订是轧差项；类别调整在没有复产 / PUD 转 PDP 数据时计 0，不许编。
- 三套价格（sec / assessment / impairment）在 `conf/price_deck.yaml` 的 `scenarios`；sec 情景的价格固定取 12 个月首日均价。
- 模板里的百分比、合计等数字必须由服务返回（如 `weight_pct`），模板里乘 100 也算智能体层做算术。

## 数据

三条轨道（合成 / 公开 / 院内真实）共用 `sql/schema.sql` 的同一套表，
每条轨道只写一个适配器放在 `src/ingest/`。

- 真实数据到位时：新增 `dqmds_adapter.py`，把源字段映射成同样的 DataFrame，**不要动上层**。
- 合成数据的真值（`data/synth_truth.csv`、`data/synth_event_truth.csv`）**只能被测试和评测用**，
  业务代码不许依赖它 —— 真实数据没有真值。
- 真实轨道的适配器除四张基础表外，还要写 `sec_unit`、`unit_well`（储量单元台账）和 `unit_plan_monthly`（计划系统）。
- 所有记录带 `data_source`；合成数据在任何对外呈现里都要标注"模拟数据"。

## 加东西的顺序

1. 先在 `src/models` / `src/reserves` / `src/sec` 里实现纯计算，配 `tests/test_kernel.py` 的测试
2. 在 `src/api/services.py` 里包一层，加 `@cached_service` 和追溯信封
3. 在 `src/agent/tools.py` 注册（记得写 description，它会进大模型的 function schema）
4. 在 `src/agent/plans.py` 的对应意图里挂上；如果是合规结论，计划里**必须**带 `search_standard`
5. 在 `src/agent/orchestrator.py::_template` 里加模板成文分支（模板必须能通过 guard）
6. 在 `src/eval/evalset.py` 加几条评测用例
7. `python -m unittest discover -s tests` 全绿，`python -m src.cli eval-agent` 不掉指标

## 测试

用 stdlib `unittest`，不引入 pytest（内网离线环境常装不上；pytest 也能直接跑这些用例）。

有几类测试不许删：

- `TestDCA::test_b_gt_1_without_dmin_raises` —— b>1 无终端递减会让 EUR 发散
- `TestDCA::test_eur_never_below_cumulative` —— EUR 小于累产是一眼假的结果
- `TestLabelingAgainstTruth` —— 标签提取能否还原合成真值，是整个仓库的地基
- `TestGuard` / `TestToolBoundary` —— 铁律与任务边界的执行者
- `TestSEC::test_checklist_citations_exist_in_corpus` —— 引用的条款必须真实存在
- `TestNewWellIdentification` / `TestMeasureEffect::test_realized_increment_against_truth` —— 新-老-措剥离能否还原合成真值
- `TestReconcileAndSensitivity` —— 对账必须闭合、敏感性方向必须对
- `TestUnitAgent` —— 单元级回答的数值一致性与条款引用

## 大模型后端

`conf/config.yaml` 的 `llm.backend` 三选一：`internal`（内网 Qwen）/ `glm`（开发对拍）/ `mock`。

**mock 不是玩具**：整条链路在没有模型时也必须能跑通并被评测。
这既是离线开发的需要，也是演示当天模型服务不可用时的兜底路径。
任何改动都不能让 `backend: mock` 跑不起来。

## 不要做的事

- 不要在 `src/agent/` 里写任何算术
- 不要为了让评测指标好看去改评测集的期望值；指标不达标就写进 README 的"已知问题"
- 不要把 `data/` 下的库文件、模型产物提交进版本库
- 不要在代码里硬编码井号、区块名、价格 —— 走 `conf/`
- 不要把真实井号、坐标写进任何示例或测试；参考材料（汇报照片、兄弟单位资料）里的真实油田名、井号、储量数字同样不许入库
- 前端的构成配色是身份色：老井基础 `--c1`、措施 `--c2`、提采新井 `--c3`、扩边井 `--c4`，全站固定；
  新增分类色必须先跑 dataviz 的 `validate_palette.js`（浅色、深色各一次）；不画双轴图
