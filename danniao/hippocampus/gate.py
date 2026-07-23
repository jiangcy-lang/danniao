"""信息触发门控引擎（向量驱动）。

输入文本 → 概念提取 → 向量嵌入 → 语义匹配主干 → 激活 / 创建主干 → 计算预测误差。

概念提取（Step 6 升级）：
- 短输入（≤8字）直接用作概念
- 长输入用 LLM 提取核心概念（1-6字）
- 概念作为主干标签和嵌入来源，原始输入存储在日志中

这模拟了人脑的感知理解——听到长句时提取核心概念，不把整句话存为记忆。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from danniao.hippocampus.tree import DynamicCognitiveTree

if TYPE_CHECKING:
    from danniao.hippocampus.concept_extractor import ConceptExtractor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GateResult:
    """门控结果。"""

    text: str
    """原始输入文本。"""

    concept: str = ""
    """提取的核心概念（主干标签来源）。"""

    matched_trunk: str | None = None
    matched_trunk_similarity: float = 0.0
    prediction_error: float = 0.0
    is_new_trunk: bool = False
    trunk_node_id: str | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _utc_now()


class InformationTriggerGate:
    """向量驱动门控：概念提取 + 语义匹配主干 + 预测误差。

    主干匹配策略：
    1. 输入文本 → 概念提取（短文本直接用，长文本用 LLM 提炼）
    2. 概念文本嵌入为向量
    3. 遍历已有主干，计算余弦相似度
    4. 最高相似度 >= TRUNK_MATCH_THRESHOLD → 匹配成功
    5. 否则 → 创建新主干（标签为概念，不是原始输入）

    预测误差：
    - 匹配到主干时：error = 1 - similarity
    - 新主干时：error = 1.0（完全意外）
    """

    TRUNK_MATCH_THRESHOLD = 0.65

    def __init__(
        self,
        tree: DynamicCognitiveTree,
        *,
        concept_extractor: ConceptExtractor | None = None,
    ) -> None:
        """初始化门控引擎。

        Args:
            tree: 认知树实例（必须已配置向量空间和嵌入管道）
            concept_extractor: 概念提取器（可选）。无则直接用原始文本。
        """
        self.tree = tree
        self.space = tree.space
        self.embedding = tree.space.embedding
        self.concept_extractor = concept_extractor

    def process(self, text: str) -> GateResult:
        """处理输入文本，执行门控决策。

        Args:
            text: 输入文本

        Returns:
            门控结果
        """
        # 0. 概念提取
        concept = text
        if self.concept_extractor:
            concept = self.concept_extractor.extract(text)

        result = GateResult(text=text, concept=concept)

        # 1. 嵌入概念文本（不是原始输入）
        input_vec = self.embedding.embed_text(concept)

        # 2. 向量匹配主干
        trunk_label, trunk_sim = self._match_trunk(input_vec)

        if trunk_label is None:
            # 3a. 未匹配 → 创建新主干（标签为概念）
            self.tree.add_trunk(concept)
            result.is_new_trunk = True
            result.matched_trunk = concept
            result.matched_trunk_similarity = 1.0
            result.prediction_error = 1.0
            result.trunk_node_id = self.tree._find_by_label(concept)
            return result

        # 3b. 匹配到 → 激活主干
        result.matched_trunk = trunk_label
        result.matched_trunk_similarity = trunk_sim
        result.prediction_error = 1.0 - trunk_sim
        result.trunk_node_id = self.tree._find_by_label(trunk_label)
        self.tree.activate(trunk_label)

        return result

    def _match_trunk(
        self, input_vec: np.ndarray
    ) -> tuple[str | None, float]:
        """用向量余弦相似度匹配已有主干。

        Args:
            input_vec: 输入概念的归一化嵌入向量

        Returns:
            (trunk_label, similarity) 或 (None, 0.0)
        """
        trunk_labels = [
            nd.get("label")
            for _, nd in self.tree.graph.nodes(data=True)
            if nd.get("kind") == "trunk" and nd.get("label")
        ]

        if not trunk_labels:
            return None, 0.0

        # 批量嵌入主干文本
        trunk_vecs = self.embedding.embed_batch(trunk_labels)

        # 已归一化向量的点积 = 余弦相似度
        similarities = trunk_vecs @ input_vec

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.TRUNK_MATCH_THRESHOLD:
            return trunk_labels[best_idx], best_sim
        return None, 0.0
