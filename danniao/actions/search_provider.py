"""可插拔搜索提供者：抽象接口与默认免 API Key 实现（阶段一）。

ABC 风格对齐 VectorStore(ABC)，让测试用手写桩替换、生产用真实 HTTP，
主心智逻辑无感知。替换为 Bing、SerpAPI、Tavily 或宿主 provider 时，
只需实现 SearchProvider / FetchProvider 接口，不改主心智。

默认实现：
- PublicSearchProvider：DuckDuckGo HTML 端点（多域名结果，免 Key）
- PublicFetchProvider：通用 HTTP 抓取 + stdlib HTML 正文抽取
- WikipediaSearchProvider / WikipediaFetchProvider：MediaWiki API（稳定 JSON）

失败策略：网络/解析异常返回空结果或 ok=False，不抛异常中断 live()。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================


@dataclass
class SearchHit:
    """单条搜索结果。"""

    url: str
    """结果链接（已还原为真实 URL）。"""

    title: str
    """结果标题。"""

    snippet: str = ""
    """结果摘要。"""

    provider: str = ""
    """返回该结果的 provider 名。"""

    rank: int = 0
    """在该 provider 结果中的位次。"""

    raw: dict = field(default_factory=dict)
    """原始数据（调试用）。"""


@dataclass
class FetchedDocument:
    """抓取到的文档。"""

    url: str
    """文档 URL。"""

    title: str = ""
    """文档标题。"""

    text: str = ""
    """正文纯文本。"""

    ok: bool = True
    """是否抓取成功。"""

    chars: int = 0
    """正文字符数。"""

    error: str = ""
    """失败时的错误信息。"""

    elapsed_ms: int = 0
    """抓取耗时（毫秒）。"""


# ==================== 抽象接口 ====================


class SearchProvider(ABC):
    """搜索提供者抽象接口。"""

    name: str = "abstract"

    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """执行单次搜索。

        Args:
            query: 搜索查询
            limit: 最多返回结果数

        Returns:
            SearchHit 列表（失败时返回空列表）
        """
        ...

    def search_many(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SearchHit]:
        """批量搜索：逐个 query 调 search 并合并，保留 provider/rank。

        Args:
            queries: 查询列表
            limit_per_query: 每个查询最多返回结果数

        Returns:
            合并后的 SearchHit 列表
        """
        merged: list[SearchHit] = []
        for q in queries:
            merged.extend(self.search(q, limit=limit_per_query))
        return merged


class FetchProvider(ABC):
    """文档抓取提供者抽象接口。"""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, url: str) -> FetchedDocument:
        """抓取单个 URL 的正文文本。

        Args:
            url: 待抓取 URL

        Returns:
            FetchedDocument（失败时 ok=False）
        """
        ...


# ==================== HTML 解析工具 ====================


class _HTMLTextExtractor(HTMLParser):
    """从 HTML 中提取纯文本，跳过脚本/样式/导航等非正文标签。

    纯标准库实现，不依赖 beautifulsoup4/lxml。
    """

    _SKIP_TAGS: frozenset[str] = frozenset({
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "header",
        "svg",
        "iframe",
    })

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        """返回提取的纯文本，连续空白折叠为单个空格。"""
        raw = " ".join(self._parts)
        # 折叠连续空白
        result: list[str] = []
        prev_space = False
        for ch in raw:
            if ch.isspace():
                if not prev_space:
                    result.append(" ")
                    prev_space = True
            else:
                result.append(ch)
                prev_space = False
        return "".join(result).strip()


class _DDGResultParser(HTMLParser):
    """解析 DuckDuckGo HTML 搜索结果页。

    DuckDuckGo HTML 端点返回的每个结果包含：
    - <a class="result__a" href="//duckduckgo.com/l/?uddg=<encoded_url>">Title</a>
    - <a class="result__snippet" ...>Snippet text</a>
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchHit] = []
        self._current_url: str = ""
        self._current_title: str = ""
        self._current_snippet: str = ""
        self._in_result_a: bool = False
        self._in_snippet: bool = False
        self._rank: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        classes = attr_dict.get("class", "") or ""

        if tag.lower() == "a" and "result__a" in classes:
            self._in_result_a = True
            self._current_url = ""
            self._current_title = ""
            self._current_snippet = ""
            href = attr_dict.get("href", "") or ""
            self._current_url = self._extract_real_url(href)

        elif tag.lower() == "a" and "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._in_result_a:
            self._in_result_a = False
            # 结果链接关闭时暂不提交，等 snippet 也收集完
            # 但如果没有 snippet，也需要提交

        elif tag.lower() == "a" and self._in_snippet:
            self._in_snippet = False
            # snippet 关闭 → 提交结果
            if self._current_url:
                self._rank += 1
                self.results.append(SearchHit(
                    url=self._current_url,
                    title=self._current_title.strip(),
                    snippet=self._current_snippet.strip(),
                    provider="duckduckgo_html",
                    rank=self._rank,
                ))
                self._current_url = ""
                self._current_title = ""
                self._current_snippet = ""

    def handle_data(self, data: str) -> None:
        if self._in_result_a:
            self._current_title += data
        elif self._in_snippet:
            self._current_snippet += data

    @staticmethod
    def _extract_real_url(href: str) -> str:
        """从 DuckDuckGo 重定向链接中还原真实 URL。

        DuckDuckGo 的结果链接形如：
        //duckduckgo.com/l/?uddg=<urlencoded_real_url>&rut=...

        提取 uddg 参数并 URL 解码得到真实 URL。
        """
        if not href:
            return ""

        # 补全协议
        if href.startswith("//"):
            href = "https:" + href

        parsed = urlparse(href)
        if parsed.path.startswith("/l/"):
            qs = parse_qs(parsed.query)
            uddg_list = qs.get("uddg", [])
            if uddg_list:
                return uddg_list[0]

        # 非重定向链接，直接返回
        return href


# ==================== 默认 Provider 实现 ====================


class PublicSearchProvider(SearchProvider):
    """DuckDuckGo HTML 搜索提供者（免 API Key）。

    使用 DuckDuckGo HTML 端点，返回多域名网页结果。
    POST 请求带 UA 头，避免被拒。失败时返回空列表。
    """

    name = "duckduckgo_html"

    DDG_URL = "https://html.duckduckgo.com/html/"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self._timeout = timeout
        self._session = session or requests.Session()

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """执行 DuckDuckGo HTML 搜索。

        Args:
            query: 搜索查询
            limit: 最多返回结果数

        Returns:
            SearchHit 列表（失败时返回空列表）
        """
        try:
            resp = self._session.post(
                self.DDG_URL,
                data={"q": query},
                headers={"User-Agent": self.USER_AGENT},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.warning(
                    "DuckDuckGo 搜索返回 HTTP %d for query=%r",
                    resp.status_code,
                    query,
                )
                return []

            parser = _DDGResultParser()
            parser.feed(resp.text)
            results = parser.results[:limit]
            return results

        except Exception as exc:
            logger.warning("DuckDuckGo 搜索失败 query=%r: %s", query, exc)
            return []


class PublicFetchProvider(FetchProvider):
    """通用 HTTP 文档抓取提供者。

    使用 requests.get 抓取网页，用标准库 HTMLParser 提取纯文本。
    失败时返回 FetchedDocument(ok=False)。
    """

    name = "generic"

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        char_limit: int = 4000,
        session: requests.Session | None = None,
    ) -> None:
        self._timeout = timeout
        self._char_limit = char_limit
        self._session = session or requests.Session()

    def fetch(self, url: str) -> FetchedDocument:
        """抓取单个 URL 的正文文本。

        Args:
            url: 待抓取 URL

        Returns:
            FetchedDocument（失败时 ok=False）
        """
        start = time.monotonic()
        try:
            resp = self._session.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=self._timeout,
            )
            elapsed = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return FetchedDocument(
                    url=url,
                    ok=False,
                    error=f"HTTP {resp.status_code}",
                    elapsed_ms=elapsed,
                )

            # 从 HTML 提取标题和正文
            title = self._extract_title(resp.text)
            extractor = _HTMLTextExtractor()
            extractor.feed(resp.text)
            text = extractor.get_text()

            # 截断到字符上限
            if len(text) > self._char_limit:
                text = text[: self._char_limit]

            return FetchedDocument(
                url=url,
                title=title,
                text=text,
                ok=True,
                chars=len(text),
                elapsed_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return FetchedDocument(
                url=url,
                ok=False,
                error=str(exc),
                elapsed_ms=elapsed,
            )

    @staticmethod
    def _extract_title(html: str) -> str:
        """从 HTML 中提取 <title> 标签内容。"""
        lower = html.lower()
        start_tag = lower.find("<title")
        if start_tag == -1:
            return ""
        content_start = lower.find(">", start_tag)
        if content_start == -1:
            return ""
        end_tag = lower.find("</title>", content_start)
        if end_tag == -1:
            return ""
        return html[content_start + 1 : end_tag].strip()


class WikipediaSearchProvider(SearchProvider):
    """Wikipedia MediaWiki API 搜索提供者（免 API Key，稳定 JSON）。

    单一域名源，单独使用永远只能到 unverified。
    建议与 PublicSearchProvider 组合使用以满足 distinct_domains >= 2 门槛。
    """

    name = "wikipedia"

    def __init__(
        self,
        *,
        lang: str = "en",
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self._lang = lang
        self._timeout = timeout
        self._session = session or requests.Session()
        self._api_url = f"https://{lang}.wikipedia.org/w/api.php"

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """执行 Wikipedia 搜索。

        Args:
            query: 搜索查询
            limit: 最多返回结果数

        Returns:
            SearchHit 列表（失败时返回空列表）
        """
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
                "format": "json",
            }
            resp = self._session.get(
                self._api_url,
                params=params,
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Wikipedia 搜索返回 HTTP %d for query=%r",
                    resp.status_code,
                    query,
                )
                return []

            data: dict[str, Any] = resp.json()
            search_results = data.get("query", {}).get("search", [])
            hits: list[SearchHit] = []
            for i, item in enumerate(search_results):
                title = item.get("title", "")
                page_url = f"https://{self._lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                snippet = item.get("snippet", "")
                hits.append(SearchHit(
                    url=page_url,
                    title=title,
                    snippet=snippet,
                    provider="wikipedia",
                    rank=i + 1,
                ))
            return hits

        except Exception as exc:
            logger.warning("Wikipedia 搜索失败 query=%r: %s", query, exc)
            return []


class WikipediaFetchProvider(FetchProvider):
    """Wikipedia REST summary 端点抓取提供者。

    直接返回纯文本 extract，无需 HTML 解析。
    """

    name = "wikipedia"

    def __init__(
        self,
        *,
        lang: str = "en",
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self._lang = lang
        self._timeout = timeout
        self._session = session or requests.Session()
        self._rest_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"

    def fetch(self, url: str) -> FetchedDocument:
        """抓取 Wikipedia 页面摘要。

        从 Wikipedia URL 中提取页面标题，调用 REST summary 端点。

        Args:
            url: Wikipedia 页面 URL

        Returns:
            FetchedDocument（失败时 ok=False）
        """
        start = time.monotonic()
        try:
            # 从 URL 提取页面标题
            parsed = urlparse(url)
            path = parsed.path
            # /wiki/Page_Title → Page_Title
            if "/wiki/" in path:
                title = path.split("/wiki/")[-1]
            else:
                title = path.rsplit("/", 1)[-1]

            if not title:
                return FetchedDocument(
                    url=url,
                    ok=False,
                    error="无法从 URL 提取 Wikipedia 页面标题",
                )

            api_url = self._rest_url + title
            resp = self._session.get(api_url, timeout=self._timeout)
            elapsed = int((time.monotonic() - start) * 1000)

            if resp.status_code != 200:
                return FetchedDocument(
                    url=url,
                    ok=False,
                    error=f"HTTP {resp.status_code}",
                    elapsed_ms=elapsed,
                )

            data: dict[str, Any] = resp.json()
            title_str = data.get("title", "")
            extract = data.get("extract", "")

            return FetchedDocument(
                url=url,
                title=title_str,
                text=extract,
                ok=True,
                chars=len(extract),
                elapsed_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return FetchedDocument(
                url=url,
                ok=False,
                error=str(exc),
                elapsed_ms=elapsed,
            )
