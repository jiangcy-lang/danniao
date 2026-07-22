"""动态认知树：主干 + 维度特征子节点；NetworkX 逻辑结构 + ChromaDB 向量双写。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from danniao.hippocampus.embeddings import Embedder, HashTextEmbedder
from danniao.hippocampus.vector_store import HippocampusVectorStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DynamicCognitiveTree:
    """渐进式认知树：树结构 + 节点 embedding（禁止纯字符串存储）。"""

    def __init__(
        self,
        *,
        vector_store: HippocampusVectorStore | None = None,
        embedder: Embedder | None = None,
        chroma_dir: str | Path = ".chroma_hippocampus",
        ephemeral_vectors: bool = False,
    ) -> None:
        self.graph = nx.DiGraph()
        self.embedder = embedder or HashTextEmbedder()
        self.vector_store = vector_store or HippocampusVectorStore(
            chroma_dir,
            ephemeral=ephemeral_vectors,
        )

    def close(self) -> None:
        self.vector_store.close()

    def _embed_label(self, label: str) -> list[float]:
        return self.embedder.encode_text(label)

    def _sync_embedding(
        self,
        node_id: str,
        *,
        concept: str,
        kind: str,
        dimension: str | None = None,
        value: str | None = None,
        parent_trunk: str | None = None,
        label_for_embed: str | None = None,
    ) -> None:
        """写入或更新向量库中的节点 embedding。"""
        text = label_for_embed or concept
        embedding = self._embed_label(text)
        self.vector_store.upsert_node(
            node_id,
            embedding,
            concept=concept,
            kind=kind,
            dimension=dimension,
            value=value,
            parent_trunk=parent_trunk,
        )
        if node_id in self.graph:
            self.graph.nodes[node_id]["embedding_id"] = node_id

    def has_embedding(self, node_id: str) -> bool:
        return self.vector_store.get_embedding(node_id) is not None

    # ---------- 写入 ----------

    def add_trunk(self, concept: str) -> dict[str, Any]:
        """仅添加孤立主干；若已存在则返回现有节点。"""
        if concept in self.graph:
            return dict(self.graph.nodes[concept])
        data = {
            "node_id": concept,
            "concept": concept,
            "kind": "trunk",
            "access_weight": 0.1,
            "creation_time": _utc_now(),
            "embedding_id": concept,
        }
        self.graph.add_node(concept, **data)
        self._sync_embedding(
            concept,
            concept=concept,
            kind="trunk",
            label_for_embed=concept,
        )
        return data

    def add_feature_child(
        self,
        trunk: str,
        dimension: str,
        value: str,
        *,
        edge_weight: float = 0.1,
    ) -> str:
        """在主干下挂载 `{dimension}-{value}` 子节点并建边。"""
        if trunk not in self.graph:
            raise KeyError(f"主干不存在: {trunk}")
        child_id = f"{dimension}-{value}"
        if child_id not in self.graph:
            self.graph.add_node(
                child_id,
                node_id=child_id,
                concept=child_id,
                kind="feature",
                dimension=dimension,
                value=value,
                access_weight=0.1,
                creation_time=_utc_now(),
                embedding_id=child_id,
            )
            self._sync_embedding(
                child_id,
                concept=child_id,
                kind="feature",
                dimension=dimension,
                value=value,
                parent_trunk=trunk,
                label_for_embed=f"{trunk} {dimension} {value}",
            )
        now = _utc_now()
        if self.graph.has_edge(trunk, child_id):
            self.graph[trunk][child_id]["weight"] = max(
                self.graph[trunk][child_id].get("weight", 0.0), edge_weight
            )
            self.graph[trunk][child_id]["last_activated_time"] = now
        else:
            self.graph.add_edge(
                trunk,
                child_id,
                weight=edge_weight,
                dimension=dimension,
                last_activated_time=now,
            )
        return child_id

    def activate(self, concept: str, *, delta: float = 0.1) -> float:
        """提升节点访问权重。"""
        if concept not in self.graph:
            raise KeyError(f"节点不存在: {concept}")
        w = float(self.graph.nodes[concept].get("access_weight", 0.0)) + delta
        self.graph.nodes[concept]["access_weight"] = w
        self.graph.nodes[concept]["last_activated_time"] = _utc_now()
        return w

    def has_feature(self, trunk: str, dimension: str, value: str) -> bool:
        child_id = f"{dimension}-{value}"
        return self.graph.has_edge(trunk, child_id)

    def match_trunk_by_vector(
        self,
        query_embedding: list[float],
        *,
        threshold: float = 0.85,
    ) -> tuple[str | None, float]:
        return self.vector_store.best_trunk_match(query_embedding, threshold=threshold)

    # ---------- 查询 / 最小表达 ----------

    def get_node(self, concept: str) -> dict[str, Any] | None:
        if concept not in self.graph:
            return None
        return dict(self.graph.nodes[concept])

    def get_children(self, trunk: str) -> list[dict[str, Any]]:
        if trunk not in self.graph:
            return []
        children: list[dict[str, Any]] = []
        for _, child, edata in self.graph.out_edges(trunk, data=True):
            nd = dict(self.graph.nodes[child])
            nd["edge_weight"] = edata.get("weight", 0.0)
            children.append(nd)
        return children

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
            val = c.get("value", c.get("concept", ""))
            parts.append(f"{dim}是{val}" if dim else str(c.get("concept")))
        return f"关于「{trunk}」，我知道：" + "、".join(parts) + "。"

    def trunk_count(self) -> int:
        return sum(1 for _, d in self.graph.nodes(data=True) if d.get("kind") == "trunk")

    def feature_count(self, trunk: str | None = None) -> int:
        if trunk is None:
            return sum(1 for _, d in self.graph.nodes(data=True) if d.get("kind") == "feature")
        return len(self.get_children(trunk))

    # ---------- 持久化 ----------

    def save_json(self, path: str) -> None:
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str, **kwargs: Any) -> DynamicCognitiveTree:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tree = cls(**kwargs)
        tree.graph = nx.node_link_graph(data, directed=True)
        return tree
