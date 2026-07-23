"""证据模型与筛选器测试（阶段一）。

测试 EvidenceFilter 的评分与分类逻辑：
accepted / disputed / unverified / noise 四态。
"""

from __future__ import annotations

from danniao.actions.evidence import (
    CandidateFact,
    EvidenceCluster,
    EvidenceFilter,
    Source,
)


# ---------- 工具函数 ----------


def _make_source(url: str, title: str = "", provider: str = "test") -> Source:
    return Source(url=url, title=title, provider=provider)


def _make_fact(
    text: str,
    kind: str = "fact",
    url: str = "https://example.com/page",
    title: str = "Example",
) -> CandidateFact:
    return CandidateFact(
        text=text,
        kind=kind,
        source=_make_source(url, title),
    )


# ==================== 评分测试 ====================


def test_score_two_domains_two_sources_no_conflict():
    """2 域名 + 2 源 + 无冲突 → confidence >= 0.5。"""
    cluster = EvidenceCluster(
        claim="Python 是解释型语言",
        supporting=[
            _make_fact("Python 是解释型语言", url="https://a.com/1"),
            _make_fact("Python 是解释型语言", url="https://b.com/1"),
        ],
        distinct_domains=2,
    )
    ef = EvidenceFilter()
    score = ef.score(cluster)
    assert score >= 0.5  # 0.3 + 0.2 = 0.5


def test_score_single_domain_penalty():
    """1 域名 → 扣 0.2。"""
    cluster = EvidenceCluster(
        claim="测试",
        supporting=[
            _make_fact("测试事实", url="https://a.com/1"),
            _make_fact("测试事实", url="https://a.com/2"),
        ],
        distinct_domains=1,
    )
    ef = EvidenceFilter()
    score = ef.score(cluster)
    # 0.3（域名≥2不触发）+ 0.2（源≥2）- 0.2（域名==1）= 0.0
    assert score == 0.0


def test_score_conflict_penalty():
    """有冲突 → 扣 0.3。"""
    cluster = EvidenceCluster(
        claim="测试",
        supporting=[
            _make_fact("测试事实一", url="https://a.com/1"),
            _make_fact("测试事实二", url="https://b.com/1"),
        ],
        conflicting=[
            _make_fact("矛盾事实", url="https://c.com/1"),
        ],
        distinct_domains=3,
    )
    ef = EvidenceFilter()
    score = ef.score(cluster)
    # 0.3 + 0.2 + 0.1 - 0.3 = 0.3
    assert abs(score - 0.3) < 0.01


def test_score_extra_domains_bonus():
    """域名 > 2 → 额外加成。"""
    cluster = EvidenceCluster(
        claim="测试",
        supporting=[
            _make_fact("事实", url="https://a.com/1"),
            _make_fact("事实", url="https://b.com/1"),
            _make_fact("事实", url="https://c.com/1"),
            _make_fact("事实", url="https://d.com/1"),
        ],
        distinct_domains=4,
    )
    ef = EvidenceFilter()
    score = ef.score(cluster)
    # 0.3 + 0.2 + 0.2(2个额外域名) = 0.7
    assert abs(score - 0.7) < 0.01


# ==================== 分类测试 ====================


def test_classify_accepted():
    """2 域名 + 2 源 + 无冲突 + confidence >= 0.5 → accepted。"""
    cluster = EvidenceCluster(
        claim="Python 是解释型语言",
        supporting=[
            _make_fact("Python 是解释型语言", url="https://a.com/1"),
            _make_fact("Python 是解释型语言", url="https://b.com/1"),
        ],
        distinct_domains=2,
    )
    ef = EvidenceFilter()
    ef.evaluate(cluster)
    assert cluster.status == "accepted"
    assert cluster.confidence >= 0.5


def test_classify_disputed():
    """有冲突 → disputed。"""
    cluster = EvidenceCluster(
        claim="地球是平的",
        supporting=[
            _make_fact("地球是平的", url="https://a.com/1"),
            _make_fact("地球是平的", url="https://b.com/1"),
        ],
        conflicting=[
            _make_fact("地球是圆的", url="https://c.com/1"),
        ],
        distinct_domains=3,
    )
    ef = EvidenceFilter()
    ef.evaluate(cluster)
    assert cluster.status == "disputed"


def test_classify_unverified():
    """1 域名 + 1 源 → unverified。"""
    cluster = EvidenceCluster(
        claim="某事物",
        supporting=[
            _make_fact("某事物是某样的", url="https://a.com/1"),
        ],
        distinct_domains=1,
    )
    ef = EvidenceFilter()
    ef.evaluate(cluster)
    assert cluster.status == "unverified"


def test_classify_noise_short_opinion():
    """单源 + opinion + 短文本 → noise。"""
    cluster = EvidenceCluster(
        claim="不错",
        supporting=[
            _make_fact("不错", kind="opinion", url="https://a.com/1"),
        ],
        distinct_domains=1,
    )
    ef = EvidenceFilter()
    ef.evaluate(cluster)
    assert cluster.status == "noise"


def test_classify_noise_marketing_words():
    """含营销词 → noise（即使长度达标）。"""
    cluster = EvidenceCluster(
        claim="限时优惠",
        supporting=[
            _make_fact(
                "这是一个限时优惠的好产品，非常值得购买",
                kind="fact",
                url="https://a.com/1",
            ),
        ],
        distinct_domains=1,
    )
    ef = EvidenceFilter()
    ef.evaluate(cluster)
    assert cluster.status == "noise"


def test_classify_noise_empty_supporting():
    """空 supporting → noise。"""
    cluster = EvidenceCluster(
        claim="空",
        supporting=[],
        distinct_domains=0,
    )
    ef = EvidenceFilter()
    ef.evaluate(cluster)
    assert cluster.status == "noise"
