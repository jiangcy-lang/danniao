"""知识吸收闭环测试（阶段一）。

测试七步闭环：查询生成、多来源对比、证据筛选、吸收门槛。
不触网，注入桩 WorldInterface。
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
    KnowledgeIngestion,
)
from danniao.actions.search_provider import FetchedDocument, SearchHit
from danniao.actions.world_interface import GatherReport
from danniao.motivation.exploration import ExplorationTarget


# ==================== 桩 WorldInterface ====================


class _StubWorld:
    """手写世界接口桩：返回预设 GatherReport。"""

    def __init__(self, report: GatherReport) -> None:
        self._report = report

    def gather(self, queries: list[str]) -> GatherReport:
        self._report.queries = queries
        return self._report


def _make_source(url: str, title: str = "") -> Source:
    return Source(url=url, title=title, provider="stub")


def _make_gather_report(
    docs: list[tuple[str, str]],
) -> GatherReport:
    """从 (url, text) 对列表构造 GatherReport。

    Args:
        docs: (url, text) 列表

    Returns:
        GatherReport
    """
    fetched = [
        FetchedDocument(url=url, text=text, ok=True, chars=len(text))
        for url, text in docs
    ]
    hits = [SearchHit(url=url, title="", snippet="", provider="stub", rank=i)
            for i, (url, _) in enumerate(docs)]
    return GatherReport(
        queries=["test"],
        hits=hits,
        fetched=fetched,
        total_chars=sum(len(text) for _, text in docs),
    )


def _make_target(text: str = "「Python」在外部世界里是怎样的？") -> ExplorationTarget:
    return ExplorationTarget(
        text=text,
        exploration_type="ask_external",
        needs_external=True,
    )


# ==================== 查询生成测试 ====================


def test_generate_queries_produces_variants():
    """查询生成产出 2-4 个变体。"""
    world = _StubWorld(GatherReport())
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    queries = ingestion._generate_queries("「Python」在外部世界里是怎样的？")

    assert 2 <= len(queries) <= 4
    assert any("Python" in q for q in queries)
    assert any("什么是" in q for q in queries)


def test_generate_queries_truncated_by_budget():
    """查询变体受 budget.max_queries 截断。"""
    from danniao.actions.world_interface import ExplorationBudget

    world = _StubWorld(GatherReport())
    budget = ExplorationBudget(max_queries=2)
    ingestion = KnowledgeIngestion(world, EvidenceFilter(), budget=budget)

    queries = ingestion._generate_queries("「Python」在外部世界里是怎样的？")

    assert len(queries) <= 2


def test_generate_queries_extracts_concept_from_brackets():
    """从「」括号中提取概念。"""
    world = _StubWorld(GatherReport())
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    queries = ingestion._generate_queries("「机器学习」在外部世界里是怎样的？")

    assert any("机器学习" in q for q in queries)


# ==================== 吸收测试 ====================


def test_accepted_when_multi_domain_multi_source():
    """多域名 + 多源 + 无冲突 → accepted → absorbed=True。"""
    docs = [
        ("https://a.com/1", "Python 是一种解释型编程语言。它由 Guido 创建。"),
        ("https://b.com/1", "Python 是一种解释型编程语言。广泛用于数据科学。"),
        ("https://c.com/1", "Python 是一种解释型编程语言。语法简洁易学。"),
    ]
    report = _make_gather_report(docs)
    world = _StubWorld(report)
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    result = ingestion.ingest(_make_target())

    assert result.absorbed is True
    assert len(result.accepted) > 0
    assert all(exp.status == "accepted" for exp in result.accepted)


def test_not_absorbed_when_single_domain():
    """单域名 → unverified → absorbed=False。"""
    docs = [
        ("https://a.com/1", "Python 是一种解释型编程语言。它由 Guido 创建。"),
        ("https://a.com/2", "Python 是一种解释型编程语言。广泛用于数据科学。"),
    ]
    report = _make_gather_report(docs)
    world = _StubWorld(report)
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    result = ingestion.ingest(_make_target())

    assert result.absorbed is False
    assert len(result.accepted) == 0


def test_disputed_when_conflict():
    """存在否定矛盾 → disputed → 不吸收。"""
    docs = [
        ("https://a.com/1", "Python 是编译型语言。"),
        ("https://b.com/1", "Python 不是编译型语言。Python 是解释型语言。"),
    ]
    report = _make_gather_report(docs)
    world = _StubWorld(report)
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    result = ingestion.ingest(_make_target())

    # 冲突的簇应归为 disputed
    assert len(result.disputed) > 0
    assert result.absorbed is False


def test_empty_results_not_absorbed():
    """无搜索结果 → absorbed=False。"""
    report = GatherReport(queries=["test"])
    world = _StubWorld(report)
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    result = ingestion.ingest(_make_target())

    assert result.absorbed is False
    assert len(result.accepted) == 0
    assert len(result.clusters) == 0


def test_noise_filtered():
    """营销内容 → noise → 不吸收。"""
    docs = [
        ("https://a.com/1", "Python 课程限时优惠，立即购买可享折扣。"),
        ("https://b.com/1", "Python 课程限时优惠，立即购买可享折扣。"),
    ]
    report = _make_gather_report(docs)
    world = _StubWorld(report)
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    result = ingestion.ingest(_make_target())

    assert result.absorbed is False


# ==================== 事实抽取测试 ====================


def test_extract_facts_skips_short_sentences():
    """短于 8 字的句子被跳过。"""
    docs = [
        ("https://a.com/1", "短句。这是一个足够长的句子。"),
    ]
    report = _make_gather_report(docs)
    world = _StubWorld(report)
    ingestion = KnowledgeIngestion(world, EvidenceFilter())

    facts = ingestion._extract_facts(report)

    # "短句。" 只有 2 字，被跳过
    assert all(len(f.text) >= 8 for f in facts)


def test_classify_kind_detects_opinion():
    """检测 opinion 类型。"""
    kind = KnowledgeIngestion._classify_kind("我认为 Python 是最好的语言。")
    assert kind == "opinion"


def test_classify_kind_detects_method():
    """检测 method 类型。"""
    kind = KnowledgeIngestion._classify_kind("首先安装 Python，然后配置环境。")
    assert kind == "method"


def test_classify_kind_detects_example():
    """检测 example 类型。"""
    kind = KnowledgeIngestion._classify_kind("例如列表是 Python 的基本数据结构。")
    assert kind == "example"


def test_classify_kind_defaults_to_fact():
    """默认为 fact 类型。"""
    kind = KnowledgeIngestion._classify_kind("Python 由 Guido van Rossum 创建。")
    assert kind == "fact"
