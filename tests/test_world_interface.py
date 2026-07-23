"""世界接口测试（阶段一）。

测试 URL 去重、预算截断、总字符触顶、fetch 失败。
不触网，用手写桩 provider。
"""

from __future__ import annotations

from danniao.actions.search_provider import (
    FetchedDocument,
    FetchProvider,
    SearchHit,
    SearchProvider,
)
from danniao.actions.world_interface import (
    ExplorationBudget,
    WorldInterface,
)


# ==================== 桩 Provider ====================


class _StubSearchProvider(SearchProvider):
    """手写搜索桩：返回预设结果。"""

    name = "stub_search"

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = hits or []

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        return self._hits[:limit]


class _StubFetchProvider(FetchProvider):
    """手写抓取桩：返回预设文档或模拟失败。"""

    name = "stub_fetch"

    def __init__(
        self,
        text: str = "这是正文内容，包含一些信息。",
        fail_urls: set[str] | None = None,
    ) -> None:
        self._text = text
        self._fail_urls = fail_urls or set()

    def fetch(self, url: str) -> FetchedDocument:
        if url in self._fail_urls:
            return FetchedDocument(url=url, ok=False, error="模拟失败")
        return FetchedDocument(
            url=url,
            title="桩标题",
            text=self._text,
            ok=True,
            chars=len(self._text),
            elapsed_ms=10,
        )


def _make_hit(url: str, title: str = "", rank: int = 0) -> SearchHit:
    return SearchHit(url=url, title=title, snippet="", provider="stub", rank=rank)


# ==================== 去重测试 ====================


def test_dedup_same_url_different_fragment():
    """同 URL 不同 fragment 合并为一条。"""
    hits = [
        _make_hit("https://example.com/page#section1", rank=1),
        _make_hit("https://example.com/page#section2", rank=2),
    ]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider()
    world = WorldInterface(search, fetch)

    report = world.gather(["test"])
    assert len(report.hits) == 1


def test_dedup_same_url_trailing_slash():
    """同 URL 尾斜杠差异合并为一条。"""
    hits = [
        _make_hit("https://example.com/page/", rank=1),
        _make_hit("https://example.com/page", rank=2),
    ]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider()
    world = WorldInterface(search, fetch)

    report = world.gather(["test"])
    assert len(report.hits) == 1


def test_dedup_same_url_host_case():
    """同 URL host 大小写差异合并为一条。"""
    hits = [
        _make_hit("https://Example.COM/page", rank=1),
        _make_hit("https://example.com/page", rank=2),
    ]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider()
    world = WorldInterface(search, fetch)

    report = world.gather(["test"])
    assert len(report.hits) == 1


def test_dedup_removes_utm_params():
    """UTM 参数不影响去重。"""
    hits = [
        _make_hit("https://example.com/page?utm_source=google", rank=1),
        _make_hit("https://example.com/page?utm_source=bing", rank=2),
    ]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider()
    world = WorldInterface(search, fetch)

    report = world.gather(["test"])
    assert len(report.hits) == 1


# ==================== 预算截断测试 ====================


def test_budget_truncates_queries():
    """queries 超过 max_queries 被截断。"""
    search = _StubSearchProvider([_make_hit("https://a.com/1")])
    fetch = _StubFetchProvider()
    budget = ExplorationBudget(max_queries=2, max_fetches=0)
    world = WorldInterface(search, fetch, budget=budget)

    report = world.gather(["q1", "q2", "q3", "q4"])
    assert len(report.queries) == 2


def test_budget_truncates_fetches():
    """fetch 候选超过 max_fetches 被截断。"""
    hits = [
        _make_hit(f"https://a.com/{i}", rank=i) for i in range(10)
    ]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider()
    budget = ExplorationBudget(max_queries=1, max_hits_per_query=10, max_fetches=3)
    world = WorldInterface(search, fetch, budget=budget)

    report = world.gather(["test"])
    assert len(report.fetched) == 3


def test_budget_total_chars_triggers_truncation():
    """总字符达 max_total_chars 后停止 fetch 且 truncated=True。"""
    long_text = "A" * 5000  # 每页 5000 字符
    hits = [_make_hit(f"https://a.com/{i}", rank=i) for i in range(10)]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider(text=long_text)
    budget = ExplorationBudget(
        max_queries=1,
        max_hits_per_query=10,
        max_fetches=10,
        per_fetch_char_limit=4000,
        max_total_chars=6000,  # 约 1.5 页后触顶
    )
    world = WorldInterface(search, fetch, budget=budget)

    report = world.gather(["test"])
    assert report.truncated is True
    assert report.total_chars <= 6000 + 4000  # 最后一次 fetch 可能超一点


# ==================== fetch 失败测试 ====================


def test_fetch_failure_skipped_and_recorded():
    """fetch 失败：跳过并记入 skipped_urls，保留 ok=False 痕迹。"""
    hits = [
        _make_hit("https://good.com/1", rank=1),
        _make_hit("https://bad.com/1", rank=2),
    ]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider(fail_urls={"https://bad.com/1"})
    world = WorldInterface(search, fetch)

    report = world.gather(["test"])
    assert len(report.fetched) == 2  # 含失败的
    failed_docs = [d for d in report.fetched if not d.ok]
    assert len(failed_docs) == 1
    assert "https://bad.com/1" in report.skipped_urls


# ==================== 单页截断测试 ====================


def test_per_fetch_char_limit():
    """单页正文截断到 per_fetch_char_limit。"""
    long_text = "B" * 10000
    hits = [_make_hit("https://a.com/1", rank=1)]
    search = _StubSearchProvider(hits)
    fetch = _StubFetchProvider(text=long_text)
    budget = ExplorationBudget(
        max_queries=1,
        max_fetches=1,
        per_fetch_char_limit=500,
        max_total_chars=10000,
    )
    world = WorldInterface(search, fetch, budget=budget)

    report = world.gather(["test"])
    assert report.fetched[0].chars == 500
    assert len(report.fetched[0].text) == 500


# ==================== 空查询测试 ====================


def test_empty_queries_returns_empty_report():
    """空查询列表返回空报告。"""
    search = _StubSearchProvider()
    fetch = _StubFetchProvider()
    world = WorldInterface(search, fetch)

    report = world.gather([])
    assert report.queries == []
    assert report.hits == []
    assert report.fetched == []
