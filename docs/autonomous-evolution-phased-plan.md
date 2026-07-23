# 丹鸟自主成长改进计划

## Summary

- 目标：将当前"同步 REPL + 被动问答 + 单次探索意愿展示"的丹鸟，演进为"统一心智内核 + 高阶任务拆解 + 可自主冒泡 + 有限自动联网 + 逐阶段真实验收"的生命体原型。
- 实施策略：采用 `/plan` 分阶段实现。每一阶段都必须同时通过三道闸门后，才允许进入下一阶段：
  - 自动化测试通过
  - 真实终端手测通过
  - 用户确认放行
- 第一阶段范围已锁定为"最小闭环"：
  - 空闲时可自发活动
  - 会主动冒泡表达当前观察/疑问/下一步
  - 支持有限自动联网获取外部信息
  - 能在用户回来后总结离线期间的活动
- 第一阶段联网不是"搜一下就说"，而是"可插拔搜索提供者 + 多轮查询 + 多来源对比 + 初步内部论证 + 通过最低真实性门槛后才吸收"的最小知识吸收闭环。本文档将该闭环补成具体契约，覆盖接口、预算、证据筛选与阶段一验收细则。
- 前期不实现固定"双轨心智"；统一为一个主心智。在识别到长期目标时，提升为"高阶任务"，再由并行执行器按资源预算拆分多个执行单元，行为类似 agent，而不是人格分裂。

## Current State Analysis

### 入口与运行方式

- 当前 CLI 入口位于 [__main__.py](file:///d:/danniao/danniao/__main__.py)。
- 入口仍是同步 REPL：`main()` 内直接调用 `mind.process(user_input)`，没有启动异步 `live()` 主循环。
- 已修复提示符 flush 问题，但 REPL 仍然是"用户输入一次 -> 系统处理一次 -> 结束本轮"的交互模型。

### 持续心智现状

- 持续心智核心位于 [continuous_mind.py](file:///d:/danniao/danniao/mind/continuous_mind.py)。
- 已存在同步 `process()` 与异步 `live()/perceive()` 两套形态。
- 异步形态已具备四条流骨架：
  - 感知流（队列 `wait_for` 超时 0.5s）
  - 动力学流（`sleep(1.0)` 突触衰减）
  - 扩散流（`sleep(0.5)` 消费激活种子）
  - 好奇心流（`sleep(2.0)` 内稳态评估）
- 自主事件队列 `self._input_queue: asyncio.Queue[tuple[str, str]]` 已存在，由 `perceive()` 生产、`_perception_stream()` 消费，是"无 tick、事件驱动"设计的关键。但仅在 `live()` 启动后才创建，而 CLI 从未启动 `live()`。
- 空闲时内部活动仅为 `_idle_rehearsal()` 的近期节点回放（取最近 5 个节点重新激活扩散），尚不具备：
  - 高阶任务对象
  - 行动调度/并行执行
  - 外部世界动作（联网搜索/抓取）
  - 自主活动对外可见的事件流
  - 离线总结能力

### 探索与表达现状

- 探索引擎位于 [exploration.py](file:///d:/danniao/danniao/motivation/exploration.py)。
- 探索目标数据类实际名为 `ExplorationTarget`（非 `ExplorationGoal`），字段为 `text` / `target_node_ids` / `drive` / `urgency` / `exploration_type`，其中 `exploration_type` 为普通字符串（非枚举），现有取值 `depth` / `relationship` / `novelty`。
- 目前探索引擎只生成目标，不执行探索。
- 表达引擎位于 [expression.py](file:///d:/danniao/danniao/expression/expression.py) 与 [ollama_expression.py](file:///d:/danniao/danniao/expression/ollama_expression.py)。
- `ExpressionContext` 与 `ProcessResult` 解耦，由 `ContinuousMind` 从结果与内稳态构建，字段含 `curiosity` / `confidence` / `energy` / `satiety`。
- 当前 LLM prompt 被强约束为"一句话、简短、自然、带有情感、不要解释不要分析"，因此天然更像婴儿式抒情回应，而不是观察、假设、计划、复盘型表达。LLM 权重冻结、不回写认知树、不可用时回退模板。

### 记忆与验证基础

- 交互日志位于 [episodic_log.py](file:///d:/danniao/danniao/hippocampus/episodic_log.py)，SQLite 追加写、幂等，字段 `id` / `timestamp` / `text` / `source` / `metadata`。`source` 为开放字符串（约定 `user` / `search` / `creator`），`metadata` 为 JSON 文本。接口 `append(interaction_id, text, *, source="user", metadata=None)` / `replay(limit, offset)` / `count()`。
- 向量认知空间位于 [vector_space.py](file:///d:/danniao/danniao/hippocampus/vector_space.py)，双视图架构：ChromaDB 向量视图（`collection="cognitive_nodes"`，cosine）+ NetworkX 图视图，以 `node_id`（`v_<sha256前16>`）关联。
- 测试基础：pytest，命名 `test_<主题>.py`。测试用手写桩类（如 `_StubStore(VectorStore)`）与确定性 `_MockEmbedding`，不使用 `unittest.mock`；`conftest.py` 在会话开始清理 `.test_chroma_*` 与 `.test_*_memory.db`；SQLite 测试用 `path=":memory:"`，ChromaDB 测试用独立目录并 `try/finally` 清理。当前测试覆盖同步处理、表达、探索目标生成、奖励闭环，但尚未覆盖异步 live 真实行为、自主活动事件、外部搜索动作、高阶任务拆解、分阶段验收流程。
- `danniao/actions/` 目录尚不存在，联网相关模块均为第一阶段新增。

## Assumptions & Decisions

- 统一心智内核优先于"双轨心智"。前期不引入固定的工作轨/研究轨分裂模型。
- "研究"与"工作"都被视为高阶任务的不同类型，由同一个驱动核和调度器管理。
- 第一阶段必须是"最小闭环"，不是只做表达升级，也不是一次性做完整自治系统。
- 自动联网默认允许，但第一阶段采用"可插拔搜索提供者"约定，而不是写死单一搜索源：
  - 代码内定义 `SearchProvider` / `FetchProvider` 抽象（ABC，与 `VectorStore(ABC)` 风格一致）
  - 默认接入一个免 API Key 的公开搜索实现作为内置 provider
  - 后续允许替换为 Bing、SerpAPI、Tavily 或宿主提供的 provider，而不改主心智逻辑
- 第一阶段自动联网不是"一次搜索"，而是"多轮查询 -> 多来源抓取 -> 初步分析 -> 内部论证 -> 低风险吸收"的最小知识吸收闭环。内部论证全程规则化，不调用 LLM 做决策。
- 第一阶段仅允许只读外部动作：
  - 允许搜索、抓取文本信息、对比多来源
  - 不执行外部写操作
  - 不执行登录、表单提交、浏览器交互
  - 不修改用户系统环境
- 每个阶段结束后必须停下，等待用户真实验收并明确放行。
- 每阶段计划都要定义"进入条件、实现范围、自动化测试、手测脚本、验收通过标准、下一阶段解锁条件"。

## Proposed Changes

## 阶段 1：最小自主闭环

### 目标

- 让丹鸟从"被动问答"升级为"可自主活动的生命体原型"。
- 范围只覆盖：
  - 空闲自主活动
  - 主动冒泡
  - 有限自动联网（吸收闭环，非搜一下就说）
  - 用户回来后的活动总结
- 不做复杂并行高阶任务，不做本地实验执行，不做代码生成学习闭环。

### 计划修改文件

- [continuous_mind.py](file:///d:/danniao/danniao/mind/continuous_mind.py)
  - 新增自主活动主循环中的"空闲探索触发"
  - 新增自主事件队列/缓冲区
  - 新增活动摘要缓存
  - 新增对外状态查询接口：最近自主活动、待播报事件、离线总结
- [exploration.py](file:///d:/danniao/danniao/motivation/exploration.py)
  - 扩展 `exploration_type` 取值，至少增加：
    - `observe`
    - `study`
    - `ask_external`
  - 增加 `needs_external: bool` 标记，标识该目标是否需要联网
- [expression.py](file:///d:/danniao/danniao/expression/expression.py)
  - 扩展 `ExpressionContext`，支持：
    - 当前活动类型
    - 活动结果摘要
    - 是否为主动冒泡
    - 是否为离线总结
- [ollama_expression.py](file:///d:/danniao/danniao/expression/ollama_expression.py)
  - 重写 prompt 约束，从"一句话抒情"改为"观察/疑问/意图/总结"可切换模式
  - 仍保持 LLM 为"嘴巴"，不负责决定行动
- [__main__.py](file:///d:/danniao/danniao/__main__.py)
  - 将同步 REPL 升级为异步入口
  - 启动 `live()` 并在 REPL 间隙轮询自主事件
  - 新增命令：
    - `activity`：查看最近自主活动
    - `summary`：查看离线总结
    - `quiet on/off`：控制主动冒泡是否即时打印
- [hippocampus/episodic_log.py](file:///d:/danniao/danniao/hippocampus/episodic_log.py)
  - 仅扩展 `source` 约定与 `metadata` 字段约定，不改表结构
  - 记录自主事件来源：`exploration`, `search`, `idle_summary`
- 新增文件 [danniao/actions/world_interface.py](file:///d:/danniao/danniao/actions/world_interface.py)
  - 统一封装"外部世界文本获取"接口
  - 只定义只读接口：`gather(queries)`
  - 负责 provider 装配、预算控制、异常规范化、来源去重
- 新增文件 [danniao/actions/search_provider.py](file:///d:/danniao/danniao/actions/search_provider.py)
  - 定义 `SearchProvider`、`FetchProvider` 抽象（ABC）与默认 provider
  - 默认 provider 支持单轮搜索、批量查询、正文抓取
  - 必须可被手写桩替换，以便测试
- 新增文件 [danniao/actions/knowledge_ingestion.py](file:///d:/danniao/danniao/actions/knowledge_ingestion.py)
  - 实现第一阶段"搜索吸收闭环"：
    - 查询生成
    - 多来源抓取
    - 来源对比
    - 证据打分
    - 可吸收结论提炼
- 新增文件 [danniao/actions/evidence.py](file:///d:/danniao/danniao/actions/evidence.py)
  - 定义证据模型、来源模型、吸收门槛、冲突判定
- 新增测试 [test_autonomous_loop.py](file:///d:/danniao/tests/test_autonomous_loop.py)
  - 覆盖空闲时自主活动、事件产出、总结生成
- 新增测试 [test_world_interface.py](file:///d:/danniao/tests/test_world_interface.py)
  - 覆盖有限自动联网的动作选择和失败处理
- 新增测试 [test_knowledge_ingestion.py](file:///d:/danniao/tests/test_knowledge_ingestion.py)
  - 覆盖多轮查询、来源去重、证据冲突、吸收门槛

### 实现要点

- `ContinuousMind.live()` 保持为唯一长期运行核心。
- 空闲时不再只做 `_idle_rehearsal()`，而是执行：
  - 判断当前是否想探索
  - 生成一个最小探索目标
  - 若目标需要联网，则进入"知识吸收闭环"
  - 提炼成一个"自主事件"
  - 写入事件缓存与 episodic log
- 自主事件必须包含：
  - `event_type`
  - `source`
  - `goal_text`
  - `observation`
  - `next_intent`
  - `timestamp`
- CLI 中，用户不输入时也能看到丹鸟主动冒泡；若开启 `quiet`，则只缓存不实时打印。
- 当用户再次输入或执行 `summary` 时，系统需能汇总"你离开期间我做了什么"。

### 第一阶段联网契约

本节是第一阶段联网能力的实现级契约。所有数据容器沿用项目既有的 `@dataclass` + `field(default_factory=...)` 风格与 PEP 604 联合类型；抽象层沿用 `VectorStore(ABC)` 的 ABC 风格；测试用手写桩类替换 provider，不触网、不用 `unittest.mock`。

#### A. Provider 抽象与数据模型（search_provider.py）

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str = ""
    provider: str = ""          # 返回该结果的 provider 名
    rank: int = 0               # 在该 provider 结果中的位次
    raw: dict = field(default_factory=dict)


@dataclass
class FetchedDocument:
    url: str
    title: str = ""
    text: str = ""              # 正文纯文本
    ok: bool = True
    chars: int = 0
    error: str = ""
    elapsed_ms: int = 0


class SearchProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]: ...

    def search_many(
        self, queries: list[str], *, limit_per_query: int = 5
    ) -> list[SearchHit]:
        """默认实现：逐个 query 调 search 并合并，保留 provider/rank。"""
        merged: list[SearchHit] = []
        for q in queries:
            merged.extend(self.search(q, limit=limit_per_query))
        return merged


class FetchProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def fetch(self, url: str) -> FetchedDocument: ...
```

- 默认内置 `PublicSearchProvider` / `PublicFetchProvider`：免 API Key，基于公开 HTML 端点，单轮搜索、批量查询、正文抽取。
- 测试中用 `_StubSearchProvider(SearchProvider)` / `_StubFetchProvider(FetchProvider)` 替换，返回固定 `SearchHit` / `FetchedDocument`，全程不触网。
- `WorldInterface` 只依赖 `SearchProvider` / `FetchProvider` 接口，不感知具体搜索源。替换为 Bing、SerpAPI、Tavily 或宿主 provider 时，主心智逻辑不变。

#### B. WorldInterface 与预算（world_interface.py）

```python
@dataclass
class ExplorationBudget:
    max_queries: int = 4
    max_hits_per_query: int = 5
    max_fetches: int = 5
    per_fetch_char_limit: int = 4000
    max_total_chars: int = 20000
    request_timeout_s: float = 10.0


@dataclass
class GatherReport:
    queries: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    fetched: list[FetchedDocument] = field(default_factory=list)
    total_chars: int = 0
    skipped_urls: list[str] = field(default_factory=list)   # 因预算/失败/去重跳过
    truncated: bool = False                                 # 是否触发总字符上限


class WorldInterface:
    def __init__(
        self,
        search: SearchProvider,
        fetch: FetchProvider,
        *,
        budget: ExplorationBudget | None = None,
    ) -> None: ...

    def gather(self, queries: list[str]) -> GatherReport: ...
```

`gather` 流程：

1. 截断 `queries` 到 `max_queries`。
2. `search.search_many(queries, limit_per_query=max_hits_per_query)` 收集候选。
3. 去重：按归一化 URL（去 fragment、去尾斜杠、host 小写）合并；同 URL 取最高 rank 的一条。
4. 选前 `max_fetches` 个候选调用 `fetch`。
5. 每个 `FetchedDocument.text` 截断到 `per_fetch_char_limit`，`chars` 据实填写。
6. 累计 `total_chars`，达到 `max_total_chars` 后停止后续抓取并置 `truncated=True`。
7. 抓取失败（`ok=False` 或抛异常）的 URL 记入 `skipped_urls`，保留 `FetchedDocument(ok=False, error=...)` 痕迹。

#### C. 证据模型与筛选（evidence.py）

```python
@dataclass
class Source:
    url: str
    title: str
    provider: str
    snippet: str = ""


@dataclass
class CandidateFact:
    text: str                    # 抽取的事实/观点/方法/例子
    kind: str                    # fact / opinion / method / example
    source: Source
    excerpt: str = ""            # 原文片段


@dataclass
class EvidenceCluster:
    claim: str                   # 归一化结论
    supporting: list[CandidateFact] = field(default_factory=list)
    conflicting: list[CandidateFact] = field(default_factory=list)
    distinct_domains: int = 0
    confidence: float = 0.0
    status: str = "unverified"   # accepted / disputed / unverified / noise


class EvidenceFilter:
    MIN_SUPPORTING = 2
    MIN_DISTINCT_DOMAINS = 2
    ACCEPT_THRESHOLD = 0.5
    NOISE_MIN_CHARS = 40         # 低于此长度且单源 opinion -> noise

    def score(self, c: EvidenceCluster) -> float:
        s = 0.0
        if c.distinct_domains >= 2:
            s += 0.3
        if len(c.supporting) >= 2:
            s += 0.2
        if c.distinct_domains > 2:
            s += min(0.2, 0.1 * (c.distinct_domains - 2))
        if c.distinct_domains == 1:
            s -= 0.2
        if c.conflicting:
            s -= 0.3
        return max(0.0, min(1.0, s))

    def classify(self, c: EvidenceCluster) -> str:
        if self._is_noise(c):
            return "noise"
        if c.conflicting:
            return "disputed"
        if (c.distinct_domains >= self.MIN_DISTINCT_DOMAINS
                and len(c.supporting) >= self.MIN_SUPPORTING
                and c.confidence >= self.ACCEPT_THRESHOLD):
            return "accepted"
        return "unverified"
```

`_is_noise` 判定：单源 + `kind == "opinion"` + 正文短于 `NOISE_MIN_CHARS`，或含明显营销词（优惠 / 立即购买 / 点击领取 等）。

最低真实性门槛（吸收门）：

- `accepted` 须同时满足：≥2 不同域名、≥2 条支持证据、`confidence ≥ 0.5`、无未解决冲突。
- 单一来源的趣闻、营销文案、无依据观点默认归为 `unverified` 或 `noise`，不吸收。
- 结论存在明显冲突时归为 `disputed`，记录为争议点，不吸收。

#### D. 知识吸收闭环（knowledge_ingestion.py）

```python
@dataclass
class CandidateExperience:
    claim: str
    supporting_sources: list[Source]
    confidence: float
    status: str                  # accepted / disputed / unverified
    task_context: str


@dataclass
class IngestionReport:
    target: ExplorationTarget
    queries: list[str] = field(default_factory=list)
    gather: GatherReport | None = None
    clusters: list[EvidenceCluster] = field(default_factory=list)
    accepted: list[CandidateExperience] = field(default_factory=list)
    disputed: list[EvidenceCluster] = field(default_factory=list)
    unverified: list[EvidenceCluster] = field(default_factory=list)
    absorbed: bool = False       # 是否至少一条 accepted


class KnowledgeIngestion:
    def __init__(
        self,
        world: WorldInterface,
        evidence_filter: EvidenceFilter,
        *,
        budget: ExplorationBudget | None = None,
    ) -> None: ...

    def ingest(self, target: ExplorationTarget) -> IngestionReport: ...
```

`ingest` 七步：

1. 查询生成：从 `target.text` 生成 2–4 个查询变体。阶段一用规则模板（如原文、`什么是{概念}`、`{概念} 原理`、`{概念} 例子`），不调用 LLM 做决策。
2. 采集：`world.gather(queries)` 得到 `GatherReport`。
3. 抽取：从 `fetched` 正文按句子级抽取 `CandidateFact`，标注 `kind`。
4. 聚类：按 `claim` 归一化（关键词对齐）合并为 `EvidenceCluster`，统计 `distinct_domains`。
5. 一致性比对与冲突标记：同 claim 下出现相互矛盾的事实入 `conflicting`。
6. 评分与分类：`EvidenceFilter.score` + `classify` 写回每个 cluster。
7. 仅 `accepted` 的 cluster 提炼为 `CandidateExperience`，`absorbed = len(accepted) > 0`。

"初步内部论证"对应步骤 5–6，全程规则化，不调用 LLM。LLM 在本阶段只作为表达层的"嘴巴"，不参与证据判定与行动决策。

#### E. 自主事件与记忆落点

```python
@dataclass
class AutonomousEvent:
    event_type: str              # exploration / search / idle_summary
    source: str                  # 与 episodic_log.source 对齐
    goal_text: str
    observation: str
    next_intent: str
    timestamp: str               # UTC ISO，复用既有 _utc_now 风格
    ingestion: IngestionReport | None = None
```

episodic_log 落点约定（不改表结构）：

- `append(event_id, observation, source=event_source, metadata={...})`。
- `source` 扩展取值：`exploration` / `search` / `idle_summary`，与既有 `user` 并列。
- `metadata` 字段：`event_type` / `goal_text` / `next_intent` / `accepted_claims`(list[str]) / `disputed_count`(int) / `confidence_avg`(float)。

ContinuousMind 集成：

- 新增可选依赖 `world_interface` / `knowledge_ingestion`（关键字参数，默认 `None`，与 `expression_engine` 一致）。
- 新增 `_autonomous_activity()`：空闲且 `wants_to_explore()` 为真时取代/增强 `_idle_rehearsal()`：
  - `exploration_engine.propose()` 取目标；若 `needs_external` 则 `knowledge_ingestion.ingest(target)`。
  - 产出 `AutonomousEvent`，写入 `_event_buffer` 与 episodic_log。
- 新增 `_event_buffer: list[AutonomousEvent]` 与 `_activity_summary` 缓存。
- 对外查询：`recent_activity(limit)` / `pending_events()` / `offline_summary()`；`status()` 增补最近自主活动摘要。

#### F. 默认预算表

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `max_queries` | 4 | 单次自主探索最多查询变体数 |
| `max_hits_per_query` | 5 | 每个查询保留结果数 |
| `max_fetches` | 5 | 单次探索最多抓取正文数 |
| `per_fetch_char_limit` | 4000 | 单页正文截断长度 |
| `max_total_chars` | 20000 | 单次探索总摄入上限 |
| `request_timeout_s` | 10.0 | 单次请求超时 |
| 查询变体数 | 2–4 | 由查询生成器产出的区间 |

#### G. 失败与退化

- 搜索失败：记录 `AutonomousEvent(source="search", observation="搜索失败")`，当轮降低活动强度，不中断 `live()`。
- 抓取失败：跳过该 URL，保留 `FetchedDocument(ok=False)` 痕迹于 `GatherReport`。
- 结果稀少：输出"我尝试寻找，但证据还不够"，`absorbed=False`。
- 来源冲突：输出争议摘要，不形成 `accepted` 结论。
- 总字符触顶：`truncated=True`，停止后续抓取，不抛异常。

#### H. 第一阶段不做

- 不做网页交互自动化
- 不做账号态访问
- 不做外部写入
- 不做代码执行式验证
- 不让 LLM 参与证据判定与行动决策

### 第一阶段吸收模型

- 吸收单元不直接写成认知树主干，而是先落成"候选经验"对象 `CandidateExperience`，至少包含：
  - `claim`
  - `supporting_sources`
  - `confidence`
  - `status`：`accepted` / `disputed` / `unverified`
  - `task_context`
- 第一阶段仅允许把 `accepted` 状态的低风险结论写入长期活动摘要与 episodic log metadata。
- 真正的"经验层"与结构化长期存储在阶段 3 再升级；阶段 1 只做最小吸收与回顾。

### 阶段 1 验收细则

验收须经三道闸门，全部通过才解锁下一阶段。

#### 闸门一 自动化测试（pytest）

新增测试沿用既有手写桩风格（桩类继承 ABC、确定性数据、`try/finally` 清理独立目录），不触网、不用 `unittest.mock`。

`tests/test_search_provider.py`：
- `_StubSearchProvider` 返回固定 `SearchHit`，`search` 与 `search_many` 合并结果正确。
- `SearchProvider` / `FetchProvider` 可被桩类替换，主心智逻辑无感知。

`tests/test_world_interface.py`：
- 去重：同 URL 不同 fragment / 尾斜杠 / host 大小写合并为一条。
- 预算截断：`queries` 超过 `max_queries` 被截断。
- 总字符触顶：达 `max_total_chars` 后停止抓取且 `truncated=True`。
- 抓取失败：跳过并记入 `skipped_urls`，保留 `ok=False` 痕迹。

`tests/test_evidence.py`：
- 评分：2 域名 2 源无冲突 -> `confidence ≥ 0.5` -> `accepted`。
- 冲突：存在 `conflicting` -> `disputed`。
- 单源：1 域名 1 源 -> `unverified`。
- 噪声：单源 `opinion` 且短于 `NOISE_MIN_CHARS` -> `noise`。

`tests/test_knowledge_ingestion.py`：
- 查询生成产出 2–4 个变体。
- 多来源对比后仅 `accepted` 写入 `CandidateExperience`。
- 全部未达标时 `absorbed=False`，`accepted` 为空。

`tests/test_autonomous_loop.py`：
- 注入桩 `WorldInterface`，空闲时产出 `AutonomousEvent`。
- 事件写入 episodic_log（`source` 与 `metadata` 字段正确）。
- `offline_summary()` 汇总事件，输出含目标/观察/吸收结论/下一步，非单句抒情。
- `quiet on/off` 控制实时打印开关。
- 既有 `test_continuous_mind.py` 与 `test_step6_integration.py` 不回归。

通过标准：`pytest -q` 全绿，新增与回归测试均通过。

#### 闸门二 真实手测脚本

1. 在 `D:\danniao` 目录下执行 `.venv\Scripts\python.exe -m danniao` 启动。
2. 不输入任何内容，静置 30 秒，观察至少 1 次自主冒泡。
3. 其中至少 1 次活动涉及联网（事件或日志含 `search` 来源）。
4. 输入 `activity`，查看最近自主活动列表。
5. 输入 `summary`，查看结构化离线总结，包含探索目标、观察、吸收结论、下一步，不是单句抒情。
6. 输入 `quiet on` 后静置，确认不再实时打印但事件仍缓存；输入 `quiet off` 恢复实时冒泡。

通过标准：上述步骤逐项可复现，联网活动确实经过吸收闭环（有 `accepted` 或明确的"证据不够"结论），而非把单次搜索结果直接复述。

#### 闸门三 用户确认

- 用户明确认可"生命感已出现，且不是低端问答；联网是吸收而非搜一下就说"。

### 阶段 1 通过后解锁

- 解锁高阶任务对象与行动规划器

## 阶段 2：高阶任务与有限并行

### 目标

- 把"一次自主活动"升级为"可持续推进的高阶任务"。
- 允许在资源允许时，将高阶任务拆解为多个执行单元并并行推进。
- 仍不做高风险本地写操作；聚焦学习、比对、验证、总结。

### 计划修改文件

- [continuous_mind.py](file:///d:/danniao/danniao/mind/continuous_mind.py)
  - 增加高阶任务注册、生命周期管理、资源预算、并行调度入口
- 新增文件 [danniao/mind/high_level_task.py](file:///d:/danniao/danniao/mind/high_level_task.py)
  - 定义高阶任务、子任务、状态流转、预算字段
- 新增文件 [danniao/mind/task_scheduler.py](file:///d:/danniao/danniao/mind/task_scheduler.py)
  - 管理串行/并行执行策略
- [exploration.py](file:///d:/danniao/danniao/motivation/exploration.py)
  - 增加"从兴趣点生成高阶任务"的规则
- 新增文件 [danniao/actions/knowledge_digest.py](file:///d:/danniao/danniao/actions/knowledge_digest.py)
  - 负责将外部信息提炼为"经验单元"
- [episodic_log.py](file:///d:/danniao/danniao/hippocampus/episodic_log.py)
  - 扩展 metadata 约定，记录 task_id / parent_task_id / result_type
- 新增测试 [test_high_level_tasks.py](file:///d:/danniao/tests/test_high_level_tasks.py)
  - 覆盖高阶任务拆解、状态流转、并行上限

### 实现要点

- 高阶任务最小字段：
  - `task_id`
  - `kind`
  - `goal`
  - `status`
  - `priority`
  - `budget`
  - `children`
  - `artifacts`
- 第一批高阶任务类型：
  - `study_topic`
  - `compare_projects`
  - `verify_claim`
- 并行不是线程优先，而是"执行单元可并行"优先；具体实现可先基于 `asyncio.create_task()`，不强依赖 OS 线程。
- 研究任务不能阻塞用户交互；REPL 输入始终有更高响应优先级。

### 阶段 2 验收标准

- 自动化测试：
  - 高阶任务与调度器测试通过
  - 阶段 1 测试不回归
- 真实手测：
  - 激活一个兴趣点后，丹鸟能把该兴趣提升为长期任务，而不是只搜索一次
  - 能显示任务进度、下一步意图、阶段性产出
  - 在与用户互动的同时，后台仍能推进至少一个研究任务
- 用户确认：
  - 用户明确认可"已经具备 agent 式高阶任务推进能力"

### 阶段 2 通过后解锁

- 解锁验证执行与行业对标闭环

## 阶段 3：验证闭环与行业对标

### 目标

- 让丹鸟不仅"学"，而且"证"。
- 对于代码、框架、方案类兴趣点，能够：
  - 提出做法
  - 形成最小验证
  - 对比行业同类实践
  - 评估自己的成果是否符合要求

### 计划修改文件

- 新增文件 [danniao/actions/verification_runner.py](file:///d:/danniao/danniao/actions/verification_runner.py)
  - 封装受限验证动作
- 新增文件 [danniao/actions/project_comparator.py](file:///d:/danniao/danniao/actions/project_comparator.py)
  - 对标同类项目结构/能力/缺口
- 新增文件 [danniao/hippocampus/experience_store.py](file:///d:/danniao/danniao/hippocampus/experience_store.py)
  - 提炼长期经验单元
- [continuous_mind.py](file:///d:/danniao/danniao/mind/continuous_mind.py)
  - 接入"学习 -> 验证 -> 反思 -> 修正 -> 经验沉淀"闭环
- 新增测试 [test_verification_loop.py](file:///d:/danniao/tests/test_verification_loop.py)
  - 覆盖验证失败后的再尝试与经验积累

### 实现要点

- 验证闭环必须区分：
  - 外部事实
  - 自己的假设
  - 已验证结论
  - 失败案例
- 对标结果不能只是摘要，必须形成结构化差异：
  - 我做了什么
  - 行业常见做法是什么
  - 差距在哪里
  - 下一步怎么补
- 若进入代码学习方向，初版只做只读分析与受控验证，不允许无边界地改动用户项目。

### 阶段 3 验收标准

- 自动化测试：
  - 新增验证闭环测试通过
  - 之前阶段测试全部保持通过
- 真实手测：
  - 选择一个明确兴趣主题后，丹鸟能自主形成学习计划、采样外部资料、给出最小验证、再做行业对标
  - 输出不再是泛泛聊天，而是有证据、有结论、有下一步
- 用户确认：
  - 用户明确认可"已经具备初步自我进化能力"

### 阶段 3 通过后解锁

- 解锁更强的长期记忆分层与任务恢复机制

## Verification Steps

### 阶段 1 验证流程

1. 运行相关单测与回归测试。
2. 用真实 Ollama + 项目入口启动丹鸟。
3. 保持终端空闲，观察是否产生自主事件。
4. 触发 `summary`，确认离线总结真实可用。
5. 将终端输出与测试结果提交给用户确认。

### 阶段 2 验证流程

1. 运行高阶任务与调度器测试。
2. 人工激活一个兴趣点，观察是否提升为长期任务。
3. 在用户交互过程中，确认后台任务仍在推进。
4. 输出任务状态与中间产物，提交用户确认。

### 阶段 3 验证流程

1. 运行验证闭环与对标测试。
2. 选择一个主题进行真实学习/验证/对标演示。
3. 确认输出包含证据、判断、差距与下一步。
4. 由用户确认是否进入后续长期能力建设。

## Risks & Guardrails

- 风险：第一阶段若同时做"冒泡 + 联网"，复杂度容易失控。
  - 控制：限定外部世界接口为只读、低风险、少量文本采样，预算上限硬编码于 `ExplorationBudget`。
- 风险：LLM 重新获得过强自主性，退化为"嘴巴即大脑"。
  - 控制：所有行动决策与证据判定必须来自探索目标、任务调度器与 `EvidenceFilter`，不由 LLM 直接决定；联网吸收闭环的内部论证全程规则化。
- 风险：联网结果被无条件吸收，污染认知。
  - 控制：只吸收通过最低真实性门槛的 `accepted` 结论；冲突与单源内容只记录不吸收。
- 风险：后台活动干扰用户交互。
  - 控制：REPL 输入优先级最高，支持 `quiet` 模式，后台只做低频输出。
- 风险：阶段范围不断膨胀。
  - 控制：每阶段冻结范围，必须经三道闸门真实验收后才可进入下一阶段。
