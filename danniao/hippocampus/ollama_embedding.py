"""Ollama 嵌入管道：通过本地 Ollama 服务生成真实语义向量。

使用 bge-m3 模型（568M, 1024维），支持中英文等多语言。
一切在本地运行，无需互联网，无需 GPU。

与 EmbeddingPipeline（sentence-transformers）接口完全一致，可互换。
"""

from __future__ import annotations

import numpy as np
import requests


class OllamaEmbedding:
    """Ollama 嵌入管道：文本 → 归一化语义向量。

    通过 Ollama HTTP API 调用本地嵌入模型。

    用法::

        emb = OllamaEmbedding()  # 默认 bge-m3
        vec = emb.embed_text("苹果")  # 1024维归一化向量
        vecs = emb.embed_batch(["苹果", "apple"])  # (2, 1024) 矩阵
    """

    DEFAULT_MODEL = "bge-m3"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        host: str = DEFAULT_HOST,
        timeout: float = 30.0,
    ) -> None:
        """初始化 Ollama 嵌入管道。

        Args:
            model_name: Ollama 模型名（默认 bge-m3）
            host: Ollama 服务地址（默认 http://localhost:11434）
            timeout: 请求超时秒数

        Raises:
            RuntimeError: Ollama 服务不可用或模型不存在
        """
        self._model = model_name
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._dim: int | None = None  # 懒加载

        # 启动时验证连接
        self._verify_connection()

    def _verify_connection(self) -> None:
        """验证 Ollama 服务可用且模型存在。"""
        try:
            resp = requests.get(
                f"{self._host}/api/tags",
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"无法连接 Ollama 服务 ({self._host})。"
                f"请确保 Ollama 正在运行。错误: {exc}"
            ) from exc

        models = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        # Ollama 模型名可能带 :latest 后缀，匹配时需兼容
        match = (
            self._model in model_names
            or f"{self._model}:latest" in model_names
            or any(n.startswith(self._model) for n in model_names)
        )
        if not match:
            raise RuntimeError(
                f"Ollama 中未找到模型 '{self._model}'。"
                f"可用模型: {', '.join(model_names[:10])}"
                f"{'...' if len(model_names) > 10 else ''}"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """将文本编码为归一化向量（L2 norm = 1）。

        Args:
            text: 输入文本（中英文均可）

        Returns:
            归一化的 numpy 向量，shape=(dim,)，dtype=float32
        """
        resp = requests.post(
            f"{self._host}/api/embed",
            json={"model": self._model, "input": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(
                f"Ollama 嵌入返回空结果。文本: '{text}'"
            )

        vec = np.asarray(embeddings[0], dtype=np.float32)

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # 缓存维度
        if self._dim is None:
            self._dim = len(vec)

        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码文本为归一化向量。

        Args:
            texts: 输入文本列表

        Returns:
            归一化的 numpy 矩阵，shape=(len(texts), dim)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Ollama /api/embed 支持批量 input
        resp = requests.post(
            f"{self._host}/api/embed",
            json={"model": self._model, "input": texts},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama 批量嵌入返回结果数量不匹配。"
                f"期望 {len(texts)}，得到 {len(embeddings) if embeddings else 0}"
            )

        vecs = np.asarray(embeddings, dtype=np.float32)

        # L2 归一化每行
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 防除零
        vecs = vecs / norms

        # 缓存维度
        if self._dim is None:
            self._dim = vecs.shape[1]

        return vecs

    @property
    def dimension(self) -> int:
        """返回嵌入维度。"""
        if self._dim is None:
            # 触发一次嵌入来获取维度
            self.embed_text("dimension_probe")
        return self._dim  # type: ignore[return-value]
