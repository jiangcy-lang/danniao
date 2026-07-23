"""突触元可塑性：激活增强 + 全局衰减（总规范 §3）。

直接操作 VectorCognitiveSpace.graph，与向量节点 ID 配合。
"""

from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NeuroDynamicsEngine:
    """动力学引擎：赫布增强与全局衰减。

    赫布增强：同时激活的两个节点之间的边权重增加。
    全局衰减：所有边权重和节点激活按 λ 衰减，数据不删除。
    """

    def __init__(self, graph: nx.DiGraph, *, max_weight: float = 1.0, decay_rate: float = 0.99) -> None:
        """初始化动力学引擎。

        Args:
            graph: 认知空间的图（NetworkX DiGraph）
            max_weight: 边权重上限
            decay_rate: 衰减率（0~1，每轮乘以此值）
        """
        self.graph = graph
        self.max_weight = max_weight
        self.decay_rate = decay_rate

    def hebbian_reinforce(
        self,
        source: str,
        target: str,
        *,
        delta: float = 0.1,
    ) -> float:
        """赫布增强：source-target 边权重 += delta。

        如果边不存在则创建。

        Args:
            source: 源节点 ID
            target: 目标节点 ID
            delta: 权重增量

        Returns:
            新的边权重
        """
        if not self.graph.has_edge(source, target):
            self.graph.add_edge(
                source,
                target,
                weight=0.0,
                last_activated_time=_utc_now(),
            )
        current = float(self.graph[source][target].get("weight", 0.0))
        new_w = min(self.max_weight, current + delta)
        self.graph[source][target]["weight"] = new_w
        self.graph[source][target]["last_activated_time"] = _utc_now()
        return new_w

    def apply_decay(self, *, decay_rate: float | None = None) -> None:
        """全局衰减：边权重 × λ；节点 activation_weight × λ。数据不删除。

        Args:
            decay_rate: 衰减率，默认使用初始化时设定的值
        """
        rate = decay_rate if decay_rate is not None else self.decay_rate
        for _, _, data in self.graph.edges(data=True):
            data["weight"] = float(data.get("weight", 0.0)) * rate
        for _, nd in self.graph.nodes(data=True):
            nd["activation_weight"] = float(nd.get("activation_weight", 0.0)) * rate

    def weaken_edge(
        self,
        source: str,
        target: str,
        *,
        delta: float = 0.05,
    ) -> float:
        """突触弱化（LTD 雏形）：source-target 边权重 -= delta。

        与 hebbian_reinforce 对称。用于交付失败时弱化导致错误的路径。
        权重不会低于 0。

        Args:
            source: 源节点 ID
            target: 目标节点 ID
            delta: 权重减量

        Returns:
            新的边权重
        """
        if not self.graph.has_edge(source, target):
            return 0.0
        current = float(self.graph[source][target].get("weight", 0.0))
        new_w = max(0.0, current - delta)
        self.graph[source][target]["weight"] = new_w
        return new_w

    def get_edge_weight(self, source: str, target: str) -> float:
        """获取边权重。"""
        if self.graph.has_edge(source, target):
            return float(self.graph[source][target].get("weight", 0.0))
        return 0.0
