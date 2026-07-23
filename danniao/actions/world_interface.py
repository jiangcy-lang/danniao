"""统一外部世界接口：预算控制 + 去重 + 异常规范化（阶段一）。

WorldInterface 只依赖 SearchProvider / FetchProvider 抽象接口，
不感知具体搜索源。替换为 Bing、SerpAPI、Tavily 或宿主 provider 时，
主心智逻辑不变。

所有外部 I/O 只读：搜索、抓取文本。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse, urlunparse

from danniao.actions.search_provider import (
    FetchedDocument,
    FetchProvider,
    SearchHit,
    SearchProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class ExplorationBudget:
    """单次自主探索的预算上限。"""

    max_queries: int = 4
    """最多查询变体数。"""

    max_hits_per_query: int = 5
    """每个查询保留结果数。"""

    max_fetches: int = 5
    """单次探索最多抓取正文数。"""

    per_fetch_char_limit: int = 4000
    """单页正文截断长度。"""

    max_total_chars: int = 20000
    """单次探索总摄入上限。"""

    request_timeout_s: float = 10.0
    """单次请求超时（秒）。"""


@dataclass
class GatherReport:
    """采集报告：一次 gather 调用的完整结果。"""

    queries: list[str] = field(default_factory=list)
    """实际使用的查询列表（已截断到 max_queries）。"""

    hits: list[SearchHit] = field(default_factory=list)
    """搜索结果（去重后）。"""

    fetched: list[FetchedDocument] = field(default_factory=list)
    """抓取到的文档列表（含失败的 ok=False 记录）。"""

    total_chars: int = 0
    """已摄入的总字符数。"""

    skipped_urls: list[str] = field(default_factory=list)
    """因预算/失败/去重跳过的 URL。"""

    truncated: bool = False
    """是否触发总字符上限。"""


class WorldInterface:
    """统一外部世界文本获取接口。

    隔离主心智与具体 provider，集中管理预算、去重、异常规范化。
    """

    def __init__(
        self,
        search: SearchProvider,
        fetch: FetchProvider,
        *,
        budget: ExplorationBudget | None = None,
    ) -> None:
        self._search = search
        self._fetch = fetch
        self._budget = budget or ExplorationBudget()

    def gather(self, queries: list[str]) -> GatherReport:
        """执行多轮查询 + 多来源抓取。

        流程：
        1. 截断 queries 到 max_queries
        2. search.search_many 收集候选
        3. URL 归一化去重
        4. 选前 max_fetches 个候选 fetch
        5. 每页截断到 per_fetch_char_limit
        6. 累计 total_chars，达 max_total_chars 停止
        7. 失败记入 skipped_urls

        Args:
            queries: 查询变体列表

        Returns:
            GatherReport 采集报告
        """
        budget = self._budget
        report = GatherReport()

        # 1. 截断查询
        report.queries = queries[: budget.max_queries]
        if not report.queries:
            return report

        # 2. 搜索
        report.hits = self._search.search_many(
            report.queries,
            limit_per_query=budget.max_hits_per_query,
        )

        # 3. 去重
        seen_urls: set[str] = set()
        deduped_hits: list[SearchHit] = []
        for hit in report.hits:
            norm = self._normalize_url(hit.url)
            if norm in seen_urls:
                report.skipped_urls.append(hit.url)
                continue
            seen_urls.add(norm)
            deduped_hits.append(hit)
        report.hits = deduped_hits

        # 4-6. 抓取（受预算控制）
        fetch_candidates = report.hits[: budget.max_fetches]
        for hit in fetch_candidates:
            # 检查总字符预算
            if report.total_chars >= budget.max_total_chars:
                report.truncated = True
                report.skipped_urls.append(hit.url)
                continue

            # 抓取
            doc = self._fetch.fetch(hit.url)

            # 截断单页
            if doc.ok and len(doc.text) > budget.per_fetch_char_limit:
                doc.text = doc.text[: budget.per_fetch_char_limit]
                doc.chars = len(doc.text)

            report.fetched.append(doc)

            if doc.ok:
                report.total_chars += doc.chars
            else:
                report.skipped_urls.append(hit.url)

        return report

    @staticmethod
    def _normalize_url(url: str) -> str:
        """URL 归一化：去 fragment、去尾斜杠、host 小写、移除跟踪参数。

        Args:
            url: 原始 URL

        Returns:
            归一化后的 URL 字符串
        """
        if not url:
            return ""

        parsed = urlparse(url)

        # host 小写
        netloc = parsed.netloc.lower()

        # 移除跟踪参数
        filtered_pairs = [
            (k, v)
            for k, v in parse_qs(parsed.query, keep_blank_values=True).items()
            if not k.lower().startswith("utm_")
            and k.lower() not in ("ref", "ref_src", "ref_url", "source")
        ]
        # 重建 query string
        query_parts: list[str] = []
        for k, vals in filtered_pairs:
            for v in vals:
                query_parts.append(f"{k}={v}")
        query = "&".join(query_parts)

        # 去尾斜杠（根路径除外）
        path = parsed.path
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")

        # 去 fragment
        return urlunparse((
            parsed.scheme,
            netloc,
            path,
            parsed.params,
            query,
            "",  # fragment 去掉
        ))

    @staticmethod
    def _domain(url: str) -> str:
        """从 URL 提取域名（去 www. 前缀）。

        Args:
            url: URL

        Returns:
            域名字符串
        """
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
