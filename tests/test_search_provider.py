"""搜索提供者测试（阶段一）。

测试 ABC 接口、桩替换、HTML 解析工具。
不触网，不用 unittest.mock。
"""

from __future__ import annotations

import pytest

from danniao.actions.search_provider import (
    FetchedDocument,
    FetchProvider,
    PublicFetchProvider,
    PublicSearchProvider,
    SearchHit,
    SearchProvider,
    WikipediaFetchProvider,
    WikipediaSearchProvider,
    _DDGResultParser,
    _HTMLTextExtractor,
)


# ==================== ABC 接口测试 ====================


def test_search_provider_abc_cannot_instantiate():
    """SearchProvider ABC 不可直接实例化。"""
    with pytest.raises(TypeError):
        SearchProvider()  # type: ignore[abstract]


def test_fetch_provider_abc_cannot_instantiate():
    """FetchProvider ABC 不可直接实例化。"""
    with pytest.raises(TypeError):
        FetchProvider()  # type: ignore[abstract]


# ==================== 桩 Provider 测试 ====================


class _StubSearchProvider(SearchProvider):
    """手写搜索桩：返回固定结果。"""

    name = "stub_search"

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = hits or [
            SearchHit(
                url="https://example.com/page1",
                title="示例页面一",
                snippet="这是摘要一",
                provider="stub_search",
                rank=1,
            ),
            SearchHit(
                url="https://example.com/page2",
                title="示例页面二",
                snippet="这是摘要二",
                provider="stub_search",
                rank=2,
            ),
        ]

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        return self._hits[:limit]


class _StubFetchProvider(FetchProvider):
    """手写抓取桩：返回固定文档。"""

    name = "stub_fetch"

    def __init__(self, text: str = "这是正文内容。") -> None:
        self._text = text

    def fetch(self, url: str) -> FetchedDocument:
        return FetchedDocument(
            url=url,
            title="桩标题",
            text=self._text,
            ok=True,
            chars=len(self._text),
            elapsed_ms=10,
        )


def test_stub_search_provider_returns_hits():
    """桩 SearchProvider 返回固定结果。"""
    provider = _StubSearchProvider()
    hits = provider.search("test")
    assert len(hits) == 2
    assert hits[0].url == "https://example.com/page1"
    assert hits[0].provider == "stub_search"


def test_search_many_merges_results():
    """search_many 合并多个查询的结果。"""
    provider = _StubSearchProvider()
    merged = provider.search_many(["query1", "query2"], limit_per_query=2)
    assert len(merged) == 4  # 2 queries × 2 hits


def test_search_many_respects_limit():
    """search_many 遵守 limit_per_query。"""
    provider = _StubSearchProvider()
    merged = provider.search_many(["q1", "q2"], limit_per_query=1)
    assert len(merged) == 2  # 2 queries × 1 hit


def test_stub_fetch_provider_returns_document():
    """桩 FetchProvider 返回固定文档。"""
    provider = _StubFetchProvider(text="测试正文。")
    doc = provider.fetch("https://example.com/test")
    assert doc.ok is True
    assert doc.text == "测试正文。"
    assert doc.chars == 5


# ==================== HTML 解析工具测试 ====================


def test_html_text_extractor_strips_script_style():
    """_HTMLTextExtractor 剥离 script/style 标签内容。"""
    html = """
    <html>
    <head><title>测试页</title></head>
    <body>
    <script>alert('hello');</script>
    <style>body { color: red; }</style>
    <p>这是正文内容。</p>
    <nav>导航菜单</nav>
    <footer>页脚信息</footer>
    </body>
    </html>
    """
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    assert "这是正文内容" in text
    assert "alert" not in text
    assert "color: red" not in text
    assert "导航菜单" not in text
    assert "页脚信息" not in text


def test_html_text_extractor_collapses_whitespace():
    """_HTMLTextExtractor 折叠连续空白。"""
    html = "<p>  多个   空格   和\n\n换行  </p>"
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    assert "  " not in text  # 无连续空格
    assert "多个 空格 和 换行" in text


def test_ddg_result_parser_extracts_results():
    """_DDGResultParser 解析 DuckDuckGo HTML 结果。"""
    html = """
    <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1&rut=abc">
            示例页面一
        </a>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">
            这是摘要一
        </a>
    </div>
    <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fanother.com%2Fpage2&rut=def">
            示例页面二
        </a>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fanother.com%2Fpage2">
            这是摘要二
        </a>
    </div>
    """
    parser = _DDGResultParser()
    parser.feed(html)
    assert len(parser.results) == 2
    assert parser.results[0].url == "https://example.com/page1"
    assert parser.results[0].title.strip() == "示例页面一"
    assert parser.results[0].snippet.strip() == "这是摘要一"
    assert parser.results[0].provider == "duckduckgo_html"
    assert parser.results[0].rank == 1
    assert parser.results[1].url == "https://another.com/page2"
    assert parser.results[1].rank == 2


def test_ddg_result_parser_handles_non_redirect_link():
    """_DDGResultParser 处理非重定向链接。"""
    html = """
    <a class="result__a" href="https://direct-link.com/page">
        直接链接
    </a>
    <a class="result__snippet" href="https://direct-link.com/page">
        摘要
    </a>
    """
    parser = _DDGResultParser()
    parser.feed(html)
    assert len(parser.results) == 1
    assert parser.results[0].url == "https://direct-link.com/page"


def test_ddg_result_parser_empty_html():
    """_DDGResultParser 对空 HTML 返回空列表。"""
    parser = _DDGResultParser()
    parser.feed("")
    assert parser.results == []


# ==================== Provider 类属性测试 ====================


def test_provider_names():
    """各 provider 有正确的 name 属性。"""
    assert PublicSearchProvider.name == "duckduckgo_html"
    assert PublicFetchProvider.name == "generic"
    assert WikipediaSearchProvider.name == "wikipedia"
    assert WikipediaFetchProvider.name == "wikipedia"


def test_default_providers_are_subclasses():
    """默认 provider 是 ABC 的子类。"""
    assert issubclass(PublicSearchProvider, SearchProvider)
    assert issubclass(PublicFetchProvider, FetchProvider)
    assert issubclass(WikipediaSearchProvider, SearchProvider)
    assert issubclass(WikipediaFetchProvider, FetchProvider)
