"""动态认知树：主干 + 维度特征子节点，空图起步，禁止扁平预置。

向量即节点架构：节点本体 = 向量，label 是投影。
VectorCognitiveSpace 为必需依赖，不再有纯字符串回退模式。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from danniao.hippocampus.vector_space import VectorCognitiveSpace


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DynamicCognitiveTree:
    """渐进式认知树（海马体记忆底座 / 图轨）。

    节点身份 = 向量哈希，label 是人类可读投影。
    所有节点操作通过 VectorCognitiveSpace 统一引擎。
    """

    def __init__(self, space: VectorCognitiveSpace) -> None:
        """初始化认知树。

        Args:
            space: 统一向量-图认知空间（必需）。必须已配置 embedding 管道。
        """
        if space is None:
            raise ValueError("VectorCognitiveSpace 是必需的，不接受 None")
        if space.embedding is None:
            raise ValueError("VectorCognitiveSpace 必须配置 embedding 管道")
        self.space = space
        self.graph = space.graph

    @staticmethod
    def _activation_weight(node_data: dict[str, Any]) -> float:
        return float(node_data.get("activation_weight", 0.0))

    # ---------- label → node_id 映射 ----------

    def _find_by_label(self, label: str) -> str | None:
        """在图中按 label 查找节点 ID。"""
        for nid, nd in self.graph.nodes(data=True):
            if nd.get("label") == label:
                return nid
        return None

    def _find_feature_by_dim_val(
        self, trunk_id: str, dimension: str, value: str
    ) -> str | None:
        """在主干的子节点中按 dimension/value 查找。"""
        for child_id in self.graph.successors(trunk_id):
            child = self.graph.nodes[child_id]
            if child.get("dimension") == dimension and child.get("value") == value:
                return child_id
        return None

    # ---------- 写入 ----------

    def add_trunk(self, concept: str) -> dict[str, Any]:
        """添加孤立主干节点。自动嵌入文本为向量。

        Args:
            concept: 主干概念文本（如「苹果」）

        Returns:
            节点数据字典
        """
        # 检查是否已有同 label 的节点
        existing = self._find_by_label(concept)
        if existing is not None:
            return dict(self.graph.nodes[existing])

        # 自动嵌入
        embedding_vec = self.space.embedding.embed_text(concept)
        node_id = self.space.add_node(
            embedding_vec,
            label=concept,
            kind="trunk",
            modality="text",
        )
        return dict(self.graph.nodes[node_id])

    def add_feature_child(
        self,
        trunk: str,
        dimension: str,
        value: str,
        *,
        edge_weight: float = 0.1,
    ) -> str:
        """在主干下挂载 ``{dimension}-{value}`` 子节点并建边。

        Args:
            trunk: 主干 label（如「苹果」）
            dimension: 维度名（如「颜色」）
            value: 维度值（如「红」）
            edge_weight: 边权重初始值

        Returns:
            子节点 node_id
        """
        trunk_id = self._find_by_label(trunk)
        if trunk_id is None:
            raise KeyError(f"主干不存在: {trunk}")

        # 检查是否已有相同 feature
        existing_child = self._find_feature_by_dim_val(trunk_id, dimension, value)
        if existing_child is not None:
            # 已存在，仅强化边
            self._strengthen_edge(trunk_id, existing_child, edge_weight)
            return existing_child

        # 嵌入 feature 文本
        feature_label = f"{dimension}-{value}"
        embedding_vec = self.space.embedding.embed_text(feature_label)
        child_id = self.space.add_node(
            embedding_vec,
            label=feature_label,
            kind="feature",
            modality="text",
            dimension=dimension,
            value=value,
        )

        # 建边
        self.space.add_edge(
            trunk_id,
            child_id,
            weight=edge_weight,
            dimension=dimension,
        )
        return child_id

    def _strengthen_edge(self, source: str, target: str, min_weight: float) -> None:
        """强化已有边权重。"""
        if self.graph.has_edge(source, target):
            current = float(self.graph[source][target].get("weight", 0.0))
            self.graph[source][target]["weight"] = max(current, min_weight)
            self.graph[source][target]["last_activated_time"] = _utc_now()

    def activate(self, label: str, *, delta: float = 0.1) -> float:
        """提升节点 activation_weight。按 label 查找。

        Args:
            label: 节点 label
            delta: 激活增量

        Returns:
            新的激活权重
        """
        node_id = self._find_by_label(label)
        if node_id is None:
            raise KeyError(f"节点不存在: {label}")
        return self.space.activate(node_id, delta=delta)

    def activate_by_id(self, node_id: str, *, delta: float = 0.1) -> float:
        """按 node_id 直接激活节点。"""
        return self.space.activate(node_id, delta=delta)

    def has_feature(self, trunk: str, dimension: str, value: str) -> bool:
        """检查主干下是否已有该特征。按 label 查找。"""
        trunk_id = self._find_by_label(trunk)
        if trunk_id is None:
            return False
        return self._find_feature_by_dim_val(trunk_id, dimension, value) is not None

    def children_ids(self, trunk: str) -> list[str]:
        """子节点 label 列表。按 label 查找主干。"""
        trunk_id = self._find_by_label(trunk)
        if trunk_id is None:
            return []
        return [
            self.graph.nodes[cid].get("label", cid)
            for cid in self.graph.successors(trunk_id)
        ]

    # ---------- 查询 / 最小表达 ----------

    def get_node(self, label: str) -> dict[str, Any] | None:
        """按 label 获取节点数据。"""
        node_id = self._find_by_label(label)
        if node_id is None:
            return None
        node = dict(self.graph.nodes[node_id])
        node["children"] = self.children_ids(label)
        return node

    def get_children(self, trunk: str) -> list[dict[str, Any]]:
        """获取主干的子节点列表。按 label 查找。"""
        trunk_id = self._find_by_label(trunk)
        if trunk_id is None:
            return []
        children: list[dict[str, Any]] = []
        for _, child_id, edata in self.graph.out_edges(trunk_id, data=True):
            nd = dict(self.graph.nodes[child_id])
            nd["children"] = [
                self.graph.nodes[c].get("label", c)
                for c in self.graph.successors(child_id)
            ]
            nd["edge_weight"] = edata.get("weight", 0.0)
            children.append(nd)
        return children

    def print_tree(self, trunk: str | None = None, *, indent: str = "  ") -> str:
        """打印树状结构。按 label 查找。"""
        lines: list[str] = []

        def _walk(node_id: str, depth: int) -> None:
            nd = self.graph.nodes[node_id]
            aw = self._activation_weight(nd)
            label = nd.get("label", node_id)
            lines.append(f"{indent * depth}{label} (w={aw:.2f})")
            for child in self.graph.successors(node_id):
                _walk(child, depth + 1)

        if trunk is not None:
            trunk_id = self._find_by_label(trunk)
            if trunk_id is None:
                return f"[未知主干] {trunk}"
            lines.append(trunk)
            for child in self.graph.successors(trunk_id):
                _walk(child, 1)
        else:
            for nid, d in self.graph.nodes(data=True):
                if d.get("kind") == "trunk":
                    label = d.get("label", nid)
                    lines.append(label)
                    for child in self.graph.successors(nid):
                        _walk(child, 1)
        return "\n".join(lines)

    def describe(self, trunk: str) -> str:
        """最小输出：用自然语言描述当前对主干的认知。"""
        node = self.get_node(trunk)
        if node is None:
            return f"我还不认识「{trunk}」。"
        children = self.get_children(trunk)
        if not children:
            return f"我知道「{trunk}」，但还没有更细的特征。"
        parts = []
        for c in children:
            dim = c.get("dimension", "")
            val = c.get("value", c.get("label", ""))
            parts.append(f"{dim}是{val}" if dim else str(c.get("label", "")))
        return f"关于「{trunk}」，我知道：" + "、".join(parts) + "。"

    def trunk_count(self) -> int:
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get("kind") == "trunk")

    def feature_count(self, trunk: str | None = None) -> int:
        if trunk is None:
            return sum(1 for _, d in self.graph.nodes(data=True) if d.get("kind") == "feature")
        return len(self.get_children(trunk))

    # ---------- 持久化 ----------

    def save_json(self, path: str) -> None:
        data = __import__("networkx").node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str, space: VectorCognitiveSpace) -> DynamicCognitiveTree:
        """从 JSON 恢复认知树。需要传入 space 实例。"""
        import networkx as nx

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tree = cls(space)
        tree.graph = nx.node_link_graph(data, directed=True)
        return tree
