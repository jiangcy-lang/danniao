"""文本 / 多模态向量编码。"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

DEFAULT_DIM = 384


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def encode_text(self, text: str) -> list[float]:
        ...


class HashTextEmbedder(Embedder):
    """确定性轻量嵌入（无外部模型，用于测试与离线）。"""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
        vec = rng.standard_normal(self._dim)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec.tolist()


class SentenceTransformerEmbedder(Embedder):
    """可选：安装 sentence-transformers 后启用高质量文本嵌入。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()


def get_default_embedder() -> Embedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashTextEmbedder()


class MultimodalEncoder:
    """跨模态对齐：文本必选；图像可选 CLIP。"""

    def __init__(self, text_embedder: Embedder | None = None) -> None:
        self.text_embedder = text_embedder or get_default_embedder()
        self._clip = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    @property
    def dimension(self) -> int:
        return self.text_embedder.dimension

    def encode_text(self, text: str) -> list[float]:
        return self.text_embedder.encode_text(text)

    def _load_clip(self) -> bool:
        if self._clip is not None:
            return True
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model.eval()
            self._clip = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = tokenizer
            self._torch = torch
            return True
        except Exception:
            return False

    def encode_image(self, image_path: str) -> list[float]:
        if not self._load_clip():
            raise RuntimeError(
                "CLIP 未安装或未就绪。请安装: pip install open-clip-torch torch pillow"
            )
        from PIL import Image

        torch = self._torch
        image = Image.open(image_path).convert("RGB")
        tensor = self._clip_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self._clip.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    def clip_available(self) -> bool:
        return self._load_clip()
