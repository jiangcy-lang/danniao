"""统一向量-图空间引擎。

向量存储（ChromaDB）和图叠加（NetworkX）由统一引擎管理。
节点 = 向量空间中的点；边 = 注意力叠加层。
两视图通过同一 node_id 关联，保证不脱节。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import networkx as nx
import numpy as np

from danniao.hippocampus.vector_hash import hash_vector
from danniao.hippocampus.vector_store import VectorStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorCognitiveSpace:
    """统一向量-图认知空间。

    向量视图（VectorStore）回答「谁在附近」。
    图视图（NetworkX DiGraph）回答「谁被显式关联」。
    两视图共享同一 node_id，由 :meth:`add_node` 保证一致写入。
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding,
    ) -> None:
        """初始化统一认知空间。

        Args:
            vector_store: 向量存储后端（必需）。
            embedding: 嵌入管道实例（必需）。
        """
        if vector_store is None:
            raise ValueError("vector_store 是必需的")
        if embedding is None:
            raise ValueError("embedding 是必需的")
        self.vector_store: VectorStore = vector_store
        self.embedding = embedding
        self.graph: nx.DiGraph = nx.DiGraph()

        # 从持久化存储同步已有节点到内存图
        self._sync_from_store()

    def _sync_from_store(self) -> None:
        """从 ChromaDB 加载已有节点到 NetworkX 图。

        ChromaDB 是持久化的，NetworkX 图是纯内存的。
        每次启动时需要同步，否则扩散激活会引用不存在的节点。

        边信息不持久化（只存在内存图中），重启后需要重新学习。
        """
        try:
            store_count = self.vector_store.count()
        except Exception:
            return

        if store_count == 0:
            return

        # 分批从 ChromaDB 加载节点到内存图
        try:
            if not hasattr(self.vector_store, '_collection'):
                return
            col = self.vector_store._collection
            batch_size = 100
            offset = 0
            loaded = 0

            while offset < store_count:
                batch = col.get(
                    limit=batch_size,
                    offset=offset,
                    include=["embeddings", "metadatas"],
                )
                ids = batch.get("ids")
                if not ids:
                    break

                metas = batch.get("metadatas")
                embs = batch.get("embeddings")

                for i, nid in enumerate(ids):
                    meta = metas[i] if metas is not None else {}
                    emb = embs[i] if embs is not None else None
                    if emb is not None:
                        emb = np.asarray(emb, dtype=np.float32)

                    self.graph.add_node(
                        nid,
                        node_id=nid,
                        label=meta.get("label", ""),
                        kind=meta.get("kind", "concept"),
                        modality=meta.get("modality", "text"),
                        embedding=emb,
                        activation_weight=float(meta.get("activation_weight", 0.1)),
                        creation_time=meta.get("creation_time", ""),
                    )
                    loaded += 1

                offset += len(ids)

            if loaded > 0:
                print(f"[认知空间] 从持久化存储加载 {loaded} 个节点")
        except Exception as e:
            # 同步失败不阻塞启动，但打印警告便于排查
            print(f"[认知空间] 警告: 持久化数据同步失败: {e}")

    def add_node(
        self,
        embedding: np.ndarray,
        *,
        label: str | None = None,
        kind: str = "concept",
        modality: str = "text",
        **metadata: Any,
    ) -> str:
        """添加向量节点到统一空间（向量存储 + 图同时写入）。

        Args:
            embedding: 节点身份向量（必需）。
            label: 人类可读投影（可选）
            kind: 节点类型（trunk / feature / abstraction）
            modality: 来源模态（text / image / multi）
            **metadata: 附加元数据

        Returns:
            node_id（从向量哈希生成）
        """
        node_id = hash_vector(embedding)
        vec_meta = {
            "label": label or "",
            "kind": kind,
            "modality": modality,
            **metadata,
        }
        self.vector_store.upsert(node_id, embedding, metadata=vec_meta)

        now = _utc_now()
        self.graph.add_node(
            node_id,
            node_id=node_id,
            label=label,
            kind=kind,
            modality=modality,
            embedding=embedding,
            activation_weight=0.1,
            creation_time=now,
            **metadata,
        )
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        weight: float = 0.1,
        **edge_data: Any,
    ) -> None:
        """添加图边（显式关系）。"""
        now = _utc_now()
        if self.graph.has_edge(source, target):
            self.graph[source][target]["weight"] = max(
                float(self.graph[source][target].get("weight", 0.0)), weight
            )
            self.graph[source][target]["last_activated_time"] = now
        else:
            self.graph.add_edge(
                source,
                target,
                weight=weight,
                last_activated_time=now,
                **edge_data,
            )

    def find_semantic_neighbors(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        """向量邻近搜索（隐含语义关系）。"""
        return self.vector_store.search(
            embedding, top_k=top_k, threshold=threshold
        )

    def get_graph_neighbors(self, node_id: str) -> list[str]:
        """图邻居（显式学习关系）。"""
        return list(self.graph.successors(node_id))

    def get_embedding(self, node_id: str) -> np.ndarray | None:
        """从图节点获取嵌入向量。"""
        if node_id not in self.graph:
            return None
        emb = self.graph.nodes[node_id].get("embedding")
        if emb is not None:
            return np.asarray(emb, dtype=np.float32)
        return None

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """获取图中的节点数据。"""
        if node_id not in self.graph:
            return None
        node_data = dict(self.graph.nodes[node_id])
        node_data["children"] = list(self.graph.successors(node_id))
        return node_data

    def activate(self, node_id: str, *, delta: float = 0.1) -> float:
        """提升节点激活权重。"""
        if node_id not in self.graph:
            raise KeyError(f"节点不存在: {node_id}")
        current = float(self.graph.nodes[node_id].get("activation_weight", 0.0))
        new_val = current + delta
        self.graph.nodes[node_id]["activation_weight"] = new_val
        self.graph.nodes[node_id]["last_activated_time"] = _utc_now()
        return new_val
