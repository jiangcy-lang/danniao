"""证据模型与规则化筛选器（阶段一：知识吸收闭环的内部论证层）。

所有判定全程规则化，不调用 LLM。LLM 在本阶段只作为表达层的"嘴巴"，
不参与证据判定与行动决策。

证据流转：
    CandidateFact（单条事实/观点/方法/例子）
      → EvidenceCluster（按 claim 归一化聚类）
        → EvidenceFilter.score + classify
          → accepted / disputed / unverified / noise

最低真实性门槛（吸收门）：
    accepted 须同时满足：
    - >= 2 不同域名
    - >= 2 条支持证据
    - confidence >= 0.5
    - 无未解决冲突
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    """信息来源。"""

    url: str
    """来源 URL。"""

    title: str
    """来源标题。"""

    provider: str
    """返回该来源的搜索提供者名称。"""

    snippet: str = ""
    """搜索结果摘要。"""


@dataclass
class CandidateFact:
    """从抓取文档中抽取的单条候选事实。"""

    text: str
    """事实/观点/方法/例子的文本。"""

    kind: str
    """类型：fact / opinion / method / example。"""

    source: Source
    """该事实的来源。"""

    excerpt: str = ""
    """原文片段（用于追溯）。"""


@dataclass
class EvidenceCluster:
    """按 claim 归一化后的证据簇。"""

    claim: str
    """归一化结论（取簇中最长事实文本）。"""

    supporting: list[CandidateFact] = field(default_factory=list)
    """支持该结论的事实列表。"""

    conflicting: list[CandidateFact] = field(default_factory=list)
    """与该结论冲突的事实列表。"""

    distinct_domains: int = 0
    """不同来源域名数。"""

    confidence: float = 0.0
    """置信度（0~1），由 EvidenceFilter.score 写入。"""

    status: str = "unverified"
    """分类状态：accepted / disputed / unverified / noise。"""


class EvidenceFilter:
    """规则化证据筛选器：评分与分类。

    所有方法只读不写 EvidenceCluster 的 supporting/conflicting/distinct_domains，
    只写 confidence 和 status。
    """

    MIN_SUPPORTING: int = 2
    """吸收门：最少支持证据数。"""

    MIN_DISTINCT_DOMAINS: int = 2
    """吸收门：最少不同域名数。"""

    ACCEPT_THRESHOLD: float = 0.5
    """吸收门：最低置信度。"""

    NOISE_MIN_CHARS: int = 40
    """噪声判定：低于此长度且单源 opinion 视为噪声。"""

    NOISE_MARKETING_WORDS: tuple[str, ...] = (
        "优惠",
        "立即购买",
        "点击领取",
        "免费领",
        "限时",
        "折扣",
        "包邮",
        "下单",
        "优惠券",
        "促销",
    )
    """营销词集：命中则视为噪声。"""

    def score(self, cluster: EvidenceCluster) -> float:
        """计算证据簇的置信度。

        评分规则：
        - 域名 >= 2：+0.3
        - 支持证据 >= 2：+0.2
        - 域名 > 2：每多一个域名 +0.1（上限 +0.2）
        - 域名 == 1：-0.2
        - 存在冲突：-0.3

        最终值 clamp 到 [0, 1]。

        Args:
            cluster: 待评分的证据簇

        Returns:
            置信度（0~1）
        """
        s = 0.0

        if cluster.distinct_domains >= 2:
            s += 0.3

        if len(cluster.supporting) >= 2:
            s += 0.2

        if cluster.distinct_domains > 2:
            s += min(0.2, 0.1 * (cluster.distinct_domains - 2))

        if cluster.distinct_domains == 1:
            s -= 0.2

        if cluster.conflicting:
            s -= 0.3

        return max(0.0, min(1.0, s))

    def classify(self, cluster: EvidenceCluster) -> str:
        """对证据簇进行分类。

        优先级：noise → disputed → accepted → unverified。

        Args:
            cluster: 待分类的证据簇

        Returns:
            状态字符串：accepted / disputed / unverified / noise
        """
        if self._is_noise(cluster):
            return "noise"

        if cluster.conflicting:
            return "disputed"

        if (
            cluster.distinct_domains >= self.MIN_DISTINCT_DOMAINS
            and len(cluster.supporting) >= self.MIN_SUPPORTING
            and cluster.confidence >= self.ACCEPT_THRESHOLD
        ):
            return "accepted"

        return "unverified"

    def evaluate(self, cluster: EvidenceCluster) -> str:
        """评分并分类，写回 confidence 和 status。

        Args:
            cluster: 待评估的证据簇

        Returns:
            分类状态字符串
        """
        cluster.confidence = self.score(cluster)
        cluster.status = self.classify(cluster)
        return cluster.status

    def _is_noise(self, cluster: EvidenceCluster) -> bool:
        """判断是否为噪声。

        条件（任一满足）：
        - 单源 + opinion + 正文短于 NOISE_MIN_CHARS
        - 任一支持证据含营销词

        Args:
            cluster: 待判断的证据簇

        Returns:
            是否为噪声
        """
        if not cluster.supporting:
            return True

        all_facts = list(cluster.supporting) + list(cluster.conflicting)
        domains = {self._domain(f.source.url) for f in all_facts}

        # 单源 + opinion + 短文本
        if len(domains) <= 1:
            for f in cluster.supporting:
                if f.kind == "opinion" and len(f.text) < self.NOISE_MIN_CHARS:
                    return True

        # 营销词检测
        for f in cluster.supporting:
            for word in self.NOISE_MARKETING_WORDS:
                if word in f.text:
                    return True

        return False

    @staticmethod
    def _domain(url: str) -> str:
        """从 URL 提取域名（去 www. 前缀）。"""
        from urllib.parse import urlparse

        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
