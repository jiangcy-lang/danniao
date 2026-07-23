"""知识吸收闭环：多轮查询→多来源抓取→对比→论证→低风险吸收（阶段一）。

七步流程：
1. 查询生成（规则模板，2-4 变体，不调 LLM）
2. 采集（WorldInterface.gather）
3. 抽取（句子级 CandidateFact，规则判定 kind）
4. 聚类（关键词 Jaccard ≥ 0.5 归同簇）
5. 冲突标记（否定/数值矛盾 → conflicting）
6. 评分分类（EvidenceFilter.score + classify）
7. 提炼（仅 accepted → CandidateExperience）

全程规则化，LLM 不参与证据判定与行动决策。
LLM 在本阶段只作为表达层的"嘴巴"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from danniao.actions.evidence import (
    CandidateFact,
    EvidenceCluster,
    EvidenceFilter,
    Source,
)
from danniao.actions.world_interface import (
    ExplorationBudget,
    GatherReport,
    WorldInterface,
)
from danniao.motivation.exploration import ExplorationTarget


# ==================== 数据模型 ====================


@dataclass
class CandidateExperience:
    """通过吸收门的候选经验。"""

    claim: str
    """归一化结论。"""

    supporting_sources: list[Source]
    """支持该结论的来源列表。"""

    confidence: float
    """置信度（0~1）。"""

    status: str
    """状态：accepted / disputed / unverified。"""

    task_context: str
    """任务上下文（探索目标文本）。"""


@dataclass
class IngestionReport:
    """知识吸收报告。"""

    target: ExplorationTarget
    """探索目标。"""

    queries: list[str] = field(default_factory=list)
    """实际使用的查询列表。"""

    gather: GatherReport | None = None
    """采集报告。"""

    clusters: list[EvidenceCluster] = field(default_factory=list)
    """所有证据簇（含已分类）。"""

    accepted: list[CandidateExperience] = field(default_factory=list)
    """通过吸收门的候选经验。"""

    disputed: list[EvidenceCluster] = field(default_factory=list)
    """争议簇（不吸收）。"""

    unverified: list[EvidenceCluster] = field(default_factory=list)
    """未验证簇（不吸收）。"""

    absorbed: bool = False
    """是否至少一条 accepted。"""


# ==================== 停用词与规则常量 ====================

# 中文停用词（单字/双字）
_STOP_WORDS: frozenset[str] = frozenset({
    "的", "是", "在", "了", "有", "和", "与", "也", "都", "就",
    "这", "那", "一", "个", "种", "等", "为", "以", "可", "能",
    "会", "对", "被", "把", "让", "使", "到", "从", "向", "于",
    "由", "并", "而", "但", "或", "且", "如", "若", "因", "所",
    "它", "他", "她", "我", "你", "们", "上", "下", "中", "里",
    "外", "前", "后", "又", "再", "还", "已", "将", "正", "刚",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "it", "this", "that", "these",
    "those", "has", "have", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "may", "might", "must",
})

# 否定词（用于冲突检测）
_NEGATION_WORDS: frozenset[str] = frozenset({
    "不", "否", "错", "无", "没", "未", "非", "别", "勿", "莫",
    "not", "no", "never", "none", "nothing", "neither", "nor",
})

# opinion 触发词
_OPINION_WORDS: frozenset[str] = frozenset({
    "应该", "认为", "觉得", "以为", "建议", "也许", "可能",
    "should", "think", "believe", "feel", "suggest", "maybe",
})

# method 触发词
_METHOD_WORDS: frozenset[str] = frozenset({
    "步骤", "方法", "首先", "然后", "接着", "最后", "流程",
    "step", "method", "first", "then", "next", "finally", "process",
})

# example 触发词
_EXAMPLE_WORDS: frozenset[str] = frozenset({
    "例如", "比如", "如", "举例", "实例", "案例",
    "example", "for instance", "such as", "like",
})

# 句子分隔符
_SENTENCE_SPLIT_RE = re.compile(r"[。！？.!?\n]+")


# ==================== 知识吸收引擎 ====================


class KnowledgeIngestion:
    """知识吸收闭环引擎。

    实现七步吸收流程，全程规则化，不调用 LLM。
    """

    def __init__(
        self,
        world: WorldInterface,
        evidence_filter: EvidenceFilter,
        *,
        budget: ExplorationBudget | None = None,
    ) -> None:
        self._world = world
        self._filter = evidence_filter
        self._budget = budget or ExplorationBudget()

    def ingest(self, target: ExplorationTarget) -> IngestionReport:
        """执行七步知识吸收闭环。

        Args:
            target: 探索目标

        Returns:
            IngestionReport 吸收报告
        """
        report = IngestionReport(target=target)

        # 步骤 1：查询生成
        report.queries = self._generate_queries(target.text)

        # 步骤 2：采集
        report.gather = self._world.gather(report.queries)

        # 步骤 3：抽取
        facts = self._extract_facts(report.gather)

        # 步骤 4：聚类
        report.clusters = self._cluster_facts(facts)

        # 步骤 5：冲突标记
        for cluster in report.clusters:
            self._mark_conflicts(cluster)

        # 步骤 6：评分分类
        for cluster in report.clusters:
            self._populate_domains(cluster, report.gather)
            self._filter.evaluate(cluster)

        # 步骤 7：提炼
        for cluster in report.clusters:
            if cluster.status == "accepted":
                report.accepted.append(self._to_experience(cluster, target.text))
            elif cluster.status == "disputed":
                report.disputed.append(cluster)
            elif cluster.status == "unverified":
                report.unverified.append(cluster)

        report.absorbed = len(report.accepted) > 0
        return report

    # ---------- 步骤 1：查询生成 ----------

    def _generate_queries(self, text: str) -> list[str]:
        """从探索目标文本生成 2-4 个查询变体。

        规则模板：原文、什么是{概念}、{概念}原理、{概念}例子。
        不调用 LLM。

        Args:
            text: 探索目标文本

        Returns:
            查询变体列表（受 budget.max_queries 截断）
        """
        concept = self._extract_concept(text)
        if not concept:
            concept = text.strip("？?")

        queries = [text.strip("？?")]
        queries.append(f"什么是{concept}")
        queries.append(f"{concept} 原理")
        queries.append(f"{concept} 例子")

        # 去重保序
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique.append(q)

        return unique[: self._budget.max_queries]

    @staticmethod
    def _extract_concept(text: str) -> str:
        """从探索目标文本中提取核心概念。

        剥掉「」括号内容，去掉问句词。

        Args:
            text: 探索目标文本

        Returns:
            核心概念字符串
        """
        # 提取「」内的内容
        match = re.search(r"[「『](.+?)[」』]", text)
        if match:
            return match.group(1)

        # 去掉常见问句前缀
        cleaned = text
        for prefix in ("我想了解更多关于", "有什么新东西可以学吗", "在外部世界里是怎样的"):
            cleaned = cleaned.replace(prefix, "")

        cleaned = cleaned.strip("？?。.，,的了的")
        return cleaned if cleaned else text

    # ---------- 步骤 3：抽取 ----------

    def _extract_facts(self, gather: GatherReport) -> list[CandidateFact]:
        """从抓取文档中按句子级抽取候选事实。

        Args:
            gather: 采集报告

        Returns:
            CandidateFact 列表
        """
        facts: list[CandidateFact] = []
        if not gather:
            return facts

        for doc in gather.fetched:
            if not doc.ok or not doc.text:
                continue

            source = Source(
                url=doc.url,
                title=doc.title,
                provider="",
                snippet="",
            )

            sentences = self._split_sentences(doc.text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 8:
                    continue

                kind = self._classify_kind(sentence)
                facts.append(CandidateFact(
                    text=sentence,
                    kind=kind,
                    source=source,
                    excerpt=sentence[:100],
                ))

        return facts

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按句子分隔符切分文本（中文。！？/英文.!?）。

        Args:
            text: 原始文本

        Returns:
            句子列表
        """
        parts = _SENTENCE_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _classify_kind(sentence: str) -> str:
        """规则判定事实类型。

        Args:
            sentence: 句子文本

        Returns:
            kind: fact / opinion / method / example
        """
        lower = sentence.lower()
        for word in _OPINION_WORDS:
            if word in lower:
                return "opinion"
        for word in _METHOD_WORDS:
            if word in lower:
                return "method"
        for word in _EXAMPLE_WORDS:
            if word in lower:
                return "example"
        return "fact"

    # ---------- 步骤 4：聚类 ----------

    def _cluster_facts(self, facts: list[CandidateFact]) -> list[EvidenceCluster]:
        """按 claim 归一化聚类。

        关键词集合 Jaccard 相似度 ≥ 0.5 归为同簇。
        claim 取簇中最长事实文本。

        Args:
            facts: 候选事实列表

        Returns:
            证据簇列表
        """
        if not facts:
            return []

        clusters: list[EvidenceCluster] = []
        # 每个簇的关键词集合
        cluster_keywords: list[set[str]] = []

        for fact in facts:
            fact_keywords = self._extract_keywords(fact.text)
            matched = False

            for i, (cluster, keywords) in enumerate(zip(clusters, cluster_keywords)):
                if self._jaccard(fact_keywords, keywords) >= 0.5:
                    cluster.supporting.append(fact)
                    # 更新关键词集合
                    keywords.update(fact_keywords)
                    # 更新 claim（取最长）
                    if len(fact.text) > len(cluster.claim):
                        cluster.claim = fact.text
                    matched = True
                    break

            if not matched:
                new_cluster = EvidenceCluster(
                    claim=fact.text,
                    supporting=[fact],
                )
                clusters.append(new_cluster)
                cluster_keywords.append(set(fact_keywords))

        return clusters

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """提取关键词集合（去停用词）。

        对中文按字符切分，对英文按空格切分。
        去掉停用词和单字符英文词。

        Args:
            text: 文本

        Returns:
            关键词集合
        """
        # 英文按空格切分
        words: set[str] = set()
        for word in text.lower().split():
            word = word.strip(".,;:!?\"'()[]{}「」『』")
            if word and word not in _STOP_WORDS and len(word) > 1:
                words.add(word)

        # 中文按字符切分（单字）
        for ch in text:
            ch_lower = ch.lower()
            if ch_lower not in _STOP_WORDS and not ch.isspace() and not ch.isascii():
                words.add(ch_lower)

        return words

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """计算 Jaccard 相似度。

        Args:
            a: 集合 A
            b: 集合 B

        Returns:
            Jaccard 相似度（0~1）
        """
        if not a or not b:
            return 0.0
        intersection = a & b
        union = a | b
        return len(intersection) / len(union)

    # ---------- 步骤 5：冲突标记 ----------

    def _mark_conflicts(self, cluster: EvidenceCluster) -> None:
        """标记簇内冲突。

        检测同簇内否定矛盾（含否定词与不含的互斥）。

        Args:
            cluster: 证据簇
        """
        if len(cluster.supporting) < 2:
            return

        # 检查是否有否定/肯定对立
        has_negation = False
        has_affirmation = False
        negation_facts: list[CandidateFact] = []

        for fact in cluster.supporting:
            if self._has_negation(fact.text):
                has_negation = True
                negation_facts.append(fact)
            else:
                has_affirmation = True

        # 如果同时有肯定和否定，把否定的事实移到 conflicting
        if has_negation and has_affirmation:
            for fact in negation_facts:
                cluster.supporting.remove(fact)
                cluster.conflicting.append(fact)

    @staticmethod
    def _has_negation(text: str) -> bool:
        """检测文本是否含否定词。

        Args:
            text: 文本

        Returns:
            是否含否定词
        """
        lower = text.lower()
        return any(word in lower for word in _NEGATION_WORDS)

    # ---------- 步骤 6：域名统计 ----------

    @staticmethod
    def _populate_domains(cluster: EvidenceCluster, gather: GatherReport) -> None:
        """统计簇内不同来源域名数。

        Args:
            cluster: 证据簇
            gather: 采集报告（用于域名提取）
        """
        all_facts = list(cluster.supporting) + list(cluster.conflicting)
        domains: set[str] = set()
        for fact in all_facts:
            domain = urlparse(fact.source.url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domains.add(domain)
        cluster.distinct_domains = len(domains)

    # ---------- 步骤 7：提炼 ----------

    @staticmethod
    def _to_experience(cluster: EvidenceCluster, task_context: str) -> CandidateExperience:
        """将 accepted 簇提炼为候选经验。

        Args:
            cluster: accepted 证据簇
            task_context: 任务上下文

        Returns:
            CandidateExperience 候选经验
        """
        sources = [fact.source for fact in cluster.supporting]
        return CandidateExperience(
            claim=cluster.claim,
            supporting_sources=sources,
            confidence=cluster.confidence,
            status=cluster.status,
            task_context=task_context,
        )
