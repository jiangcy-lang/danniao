"""ChromaDB 向量存储实现（第二层：活跃向量空间）。

使用 ChromaDB 的 PersistentClient + cosine 距离空间。
distance = 1 - cosine_similarity，search 方法做阈值后过滤。
"""

from __future__ import annotations

import gc
import os
import shutil
from typing import Any, Sequence

from danniao.hippocampus.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB 持久化向量存储。

    数据落盘到指定目录，启动时自动加载。
    使用 cosine 距离空间，``distance = 1 - similarity``。

    支持维度自适应：当 ``expected_dim`` 指定时，如果现有数据维度
    不匹配（如嵌入模型从 768 维换为 1024 维），自动清理旧数据并重建。
    """

    def __init__(
        self,
        path: str = ".chroma_hippocampus",
        collection_name: str = "cognitive_nodes",
        *,
        expected_dim: int | None = None,
    ) -> None:
        """初始化 ChromaDB 向量存储。

        Args:
            path: 数据落盘目录
            collection_name: 集合名
            expected_dim: 期望的向量维度。指定后，如果现有数据维度
                          不匹配，自动清理旧数据重建。
        """
        self._path = path
        self._collection_name = collection_name
        self._expected_dim = expected_dim
        self._client = None
        self._collection = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化 ChromaDB 客户端和集合。

        如果指定了 expected_dim 且目录中已有旧数据，
        检查维度是否匹配。不匹配则清理旧数据重建。
        """
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb 未安装。请运行: pip install chromadb"
            ) from exc

        os.makedirs(self._path, exist_ok=True)

        # 维度自适应：在创建正式 client 前检查旧数据维度
        if self._expected_dim is not None and self._has_existing_data():
            self._check_and_rebuild_if_needed(chromadb)

        self._client = chromadb.PersistentClient(path=self._path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _has_existing_data(self) -> bool:
        """检查目录中是否已有 ChromaDB 数据。"""
        try:
            entries = os.listdir(self._path)
            return len(entries) > 0
        except OSError:
            return False

    def _check_and_rebuild_if_needed(self, chromadb_module) -> None:
        """检查现有数据维度，不匹配则清理重建。

        使用临时 client 检查，检查完毕后立即释放，
        然后删除整个目录——避免 Windows 文件锁问题。
        """
        try:
            temp_client = chromadb_module.PersistentClient(path=self._path)
            temp_col = temp_client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            if temp_col.count() > 0:
                existing = temp_col.get(limit=1, include=["embeddings"])
                if existing.get("embeddings") and len(existing["embeddings"]) > 0:
                    existing_dim = len(existing["embeddings"][0])
                    if existing_dim != self._expected_dim:
                        # 维度不匹配 → 释放临时 client，删除目录
                        print(
                            f"[ChromaDB] 维度变更: {existing_dim} → {self._expected_dim}，"
                            f"清理旧数据重建"
                        )
                        del temp_col
                        del temp_client
                        gc.collect()
                        shutil.rmtree(self._path, ignore_errors=True)
                        os.makedirs(self._path, exist_ok=True)
                        return

            # 维度匹配 → 释放临时 client，让正式 client 接管
            del temp_col
            del temp_client
            gc.collect()
        except Exception:
            # 检查失败不阻塞初始化，让正式 client 处理
            pass

    def upsert(
        self,
        node_id: str,
        vector: Sequence[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """插入或更新向量节点。

        Args:
            node_id: 节点唯一标识
            vector: 嵌入向量
            metadata: 可选元数据（label / kind / modality 等）
        """
        # ChromaDB metadata 值必须是 str / int / float / bool / None
        clean_meta = {}
        if metadata:
            for k, v in metadata.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

        # ChromaDB 要求原生 Python float，不接受 numpy float32
        embeddings_list = [[float(x) for x in vector]]

        self._collection.upsert(
            ids=[node_id],
            embeddings=embeddings_list,
            metadatas=[clean_meta] if clean_meta else None,
            documents=[clean_meta.get("label", "")],
        )

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        """按向量相似度检索。

        ChromaDB cosine 空间下 ``distance = 1 - similarity``，
        因此 ``similarity >= threshold`` 等价于 ``distance <= 1 - threshold``。

        Args:
            vector: 查询向量
            top_k: 返回条数上限
            threshold: 相似度阈值

        Returns:
            ``[(node_id, similarity), ...]`` 按相似度降序
        """
        # 动态限制 top_k，避免 ChromaDB 警告
        actual_count = self._collection.count()
        if actual_count == 0:
            return []
        effective_k = min(top_k, actual_count)

        # ChromaDB 要求原生 Python float
        query_vec = [float(x) for x in vector]
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=effective_k,
            include=["metadatas", "distances"],
        )

        ids_list = results.get("ids", [[]])
        distances_list = results.get("distances", [[]])

        max_distance = 1.0 - threshold
        kept: list[tuple[str, float]] = []

        if not ids_list:
            return kept

        for node_id, dist in zip(ids_list[0], distances_list[0]):
            similarity = 1.0 - dist
            if similarity >= threshold:
                kept.append((node_id, similarity))

        # 按相似度降序
        kept.sort(key=lambda x: x[1], reverse=True)
        return kept

    def get(self, node_id: str) -> dict[str, Any] | None:
        """按 ID 获取节点（含向量、元数据）。"""
        results = self._collection.get(
            ids=[node_id],
            include=["embeddings", "metadatas", "documents"],
        )
        if not results["ids"]:
            return None
        entry: dict[str, Any] = {
            "id": results["ids"][0],
            "embedding": results["embeddings"][0] if results.get("embeddings") else None,
            "metadata": results["metadatas"][0] if results.get("metadatas") else {},
            "document": results["documents"][0] if results.get("documents") else "",
        }
        return entry

    def count(self) -> int:
        """返回集合中的向量总数。"""
        return self._collection.count()
