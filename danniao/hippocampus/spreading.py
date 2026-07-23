"""扩散激活：联想的核心机制（Step 4B）。

激活沿图边（强，受边权重调制）和向量邻近（弱，受余弦相似度调制）双向传播。
类比人脑的联想 —— 想到「苹果」会自然联想到「红色」「甜味」。

这是丹鸟"思考"的开始：不是被动等输入，而是激活自行扩散，
让相关概念浮现到意识中。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from danniao.hippocampus.vector_space import VectorCognitiveSpace


@dataclass
class SpreadConfig:
    """扩散激活配置参数。

    所有参数均可运行时调整，用于调谐联想强度。
    """

    edge_factor: float = 0.5
    """沿图边传播强度（显式学习关系）。"""

    vector_factor: float = 0.2
    """沿向量邻近传播强度（隐含语义关系）。"""

    decay_per_hop: float = 0.7
    """每跳衰减系数。防止无限传播。"""

    activation_threshold: float = 0.05
    """低于此值的激活不再传播。"""

    max_hops: int = 3
    """最大传播深度。"""

    top_k_vector: int = 5
    """向量邻近搜索数量。"""

    vector_threshold: float = 0.6
    """向量邻近搜索相似度阈值。"""


class SpreadingActivation:
    """扩散激活引擎。

    从种子节点出发，沿图边和向量邻近双向传播激活。
    每个节点最多被扩展一次（防止环路），激活值累加但上限 1.0。

    用法::

        spreader = SpreadingActivation(space)
        activated = spreader.spread([seed_node_id])
        # activated = {node_id: activation_level, ...}
    """

    def __init__(
        self,
        space: VectorCognitiveSpace,
        *,
        config: SpreadConfig | None = None,
    ) -> None:
        """初始化扩散激活引擎。

        Args:
            space: 统一向量-图认知空间
            config: 扩散配置（可选，使用默认值）
        """
        self.space = space
        self.config = config or SpreadConfig()

    def spread(
        self,
        seed_node_ids: list[str],
        *,
        initial_activation: float = 1.0,
    ) -> dict[str, float]:
        """从种子节点扩散激活。

        Args:
            seed_node_ids: 种子节点 ID 列表
            initial_activation: 种子节点初始激活水平（0~1）

        Returns:
            ``{node_id: activation_level}`` 所有被激活的节点及其激活水平。
            种子节点也包含在内。激活水平 0~1。
        """
        if not seed_node_ids:
            return {}

        cfg = self.config
        activation: dict[str, float] = {}
        expanded: set[str] = set()

        # 初始化种子节点
        for nid in seed_node_ids:
            if nid in self.space.graph:
                activation[nid] = initial_activation

        if not activation:
            return {}

        # 逐跳传播
        for _hop in range(cfg.max_hops):
            # 找出需要扩展的节点：激活值高于阈值且尚未扩展
            to_expand = [
                nid
                for nid, level in activation.items()
                if level >= cfg.activation_threshold and nid not in expanded
            ]

            if not to_expand:
                break

            for node_id in to_expand:
                expanded.add(node_id)
                current_level = activation[node_id]

                # 1. 沿图边传播（强）
                self._spread_along_edges(
                    node_id, current_level, activation
                )

                # 2. 沿向量邻近传播（弱）
                self._spread_along_vectors(
                    node_id, current_level, activation
                )

        return activation

    def _spread_along_edges(
        self,
        node_id: str,
        current_level: float,
        activation: dict[str, float],
    ) -> None:
        """沿图边传播激活。"""
        cfg = self.config
        edge_spread_base = current_level * cfg.edge_factor * cfg.decay_per_hop

        # 防御：节点可能不在图中（ChromaDB 持久化数据与内存图不同步）
        if node_id not in self.space.graph:
            return

        for neighbor in self.space.graph.successors(node_id):
            edge_weight = float(
                self.space.graph[node_id][neighbor].get("weight", 0.1)
            )
            spread_amount = edge_spread_base * edge_weight
            if spread_amount > 0:
                activation[neighbor] = min(
                    1.0,
                    activation.get(neighbor, 0.0) + spread_amount,
                )

    def _spread_along_vectors(
        self,
        node_id: str,
        current_level: float,
        activation: dict[str, float],
    ) -> None:
        """沿向量邻近传播激活。"""
        cfg = self.config
        vec_spread_base = current_level * cfg.vector_factor * cfg.decay_per_hop

        node_vec = self.space.get_embedding(node_id)
        if node_vec is None:
            return

        try:
            neighbors = self.space.find_semantic_neighbors(
                node_vec,
                top_k=cfg.top_k_vector,
                threshold=cfg.vector_threshold,
            )
        except Exception:
            # 向量存储不可用时静默跳过（测试环境可能无 ChromaDB）
            return

        for neighbor_id, similarity in neighbors:
            if neighbor_id == node_id:
                continue
            # 只激活图中存在的节点（ChromaDB 可能有旧数据不在内存图中）
            if neighbor_id not in self.space.graph:
                continue
            spread_amount = vec_spread_base * float(similarity)
            if spread_amount > 0:
                activation[neighbor_id] = min(
                    1.0,
                    activation.get(neighbor_id, 0.0) + spread_amount,
                )

    def get_activated_nodes(
        self,
        seed_node_ids: list[str],
        *,
        threshold: float = 0.1,
        initial_activation: float = 1.0,
    ) -> list[tuple[str, float]]:
        """扩散激活并返回按激活水平降序排列的节点列表。

        Args:
            seed_node_ids: 种子节点 ID
            threshold: 返回节点的最低激活水平
            initial_activation: 种子初始激活水平

        Returns:
            ``[(node_id, activation_level), ...]`` 按激活水平降序
        """
        activation = self.spread(
            seed_node_ids, initial_activation=initial_activation
        )
        result = [
            (nid, level)
            for nid, level in activation.items()
            if level >= threshold
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
