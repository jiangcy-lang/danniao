"""海马体：动态认知树与信息门控（记忆底座，非整脑）。

向量即节点架构（2026-07-22 重构）：
- EmbeddingPipeline：sentence-transformers 嵌入管道
- OllamaEmbedding：Ollama 本地嵌入管道（bge-m3，1024维）
- hash_vector：向量确定性哈希
- ChromaVectorStore：ChromaDB 向量存储
- VectorCognitiveSpace：统一向量-图空间
- EpisodicLog：第一层原始日志（SQLite）
- DynamicCognitiveTree：认知树（VectorCognitiveSpace 必需）
- InformationTriggerGate：向量驱动门控（不再依赖规则词典）
- SpreadingActivation：扩散激活引擎（Step 4B，联想核心）
"""

from danniao.hippocampus.chroma_store import ChromaVectorStore
from danniao.hippocampus.concept_extractor import ConceptExtractor
from danniao.hippocampus.dynamics import NeuroDynamicsEngine
from danniao.hippocampus.embeddings import EmbeddingPipeline
from danniao.hippocampus.episodic_log import EpisodicLog
from danniao.hippocampus.gate import GateResult, InformationTriggerGate
from danniao.hippocampus.ollama_embedding import OllamaEmbedding
from danniao.hippocampus.spreading import SpreadingActivation, SpreadConfig
from danniao.hippocampus.tree import DynamicCognitiveTree
from danniao.hippocampus.vector_hash import hash_vector
from danniao.hippocampus.vector_space import VectorCognitiveSpace
from danniao.hippocampus.vector_store import VectorStore

__all__ = [
    "DynamicCognitiveTree",
    "InformationTriggerGate",
    "GateResult",
    "NeuroDynamicsEngine",
    "VectorStore",
    "EmbeddingPipeline",
    "OllamaEmbedding",
    "ConceptExtractor",
    "hash_vector",
    "ChromaVectorStore",
    "VectorCognitiveSpace",
    "EpisodicLog",
    "SpreadingActivation",
    "SpreadConfig",
]
