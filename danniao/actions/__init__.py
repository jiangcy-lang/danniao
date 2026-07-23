"""丹鸟行动层：外部世界交互与知识吸收（阶段一）。

行动层封装丹鸟与外部世界的只读交互：
- 搜索（SearchProvider / FetchProvider 抽象 + 默认免 Key 实现）
- 世界接口（WorldInterface：预算控制 + 去重 + 异常规范化）
- 证据筛选（EvidenceFilter：规则化评分与分类）
- 知识吸收（KnowledgeIngestion：七步吸收闭环）

所有外部 I/O 只读：搜索、抓取文本、对比多来源。
不执行外部写操作、登录、表单提交或浏览器交互。
"""

from __future__ import annotations

from danniao.actions.evidence import (
    CandidateFact,
    EvidenceCluster,
    EvidenceFilter,
    Source,
)
from danniao.actions.knowledge_ingestion import (
    CandidateExperience,
    IngestionReport,
    KnowledgeIngestion,
)
from danniao.actions.search_provider import (
    FetchedDocument,
    FetchProvider,
    PublicFetchProvider,
    PublicSearchProvider,
    SearchHit,
    SearchProvider,
    WikipediaFetchProvider,
    WikipediaSearchProvider,
)
from danniao.actions.world_interface import (
    ExplorationBudget,
    GatherReport,
    WorldInterface,
)

__all__ = [
    # 数据模型
    "SearchHit",
    "FetchedDocument",
    "Source",
    "CandidateFact",
    "EvidenceCluster",
    "CandidateExperience",
    "IngestionReport",
    "ExplorationBudget",
    "GatherReport",
    # 抽象接口
    "SearchProvider",
    "FetchProvider",
    "EvidenceFilter",
    "WorldInterface",
    "KnowledgeIngestion",
    # 默认实现
    "PublicSearchProvider",
    "PublicFetchProvider",
    "WikipediaSearchProvider",
    "WikipediaFetchProvider",
]
