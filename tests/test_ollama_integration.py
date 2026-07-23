"""Ollama 集成测试：使用真实 Ollama 服务验证端到端链路。

这些测试需要 Ollama 运行在 localhost:11434，且安装了 nomic-embed-text 和 qwen3.5:2b。
如果 Ollama 不可用，测试自动跳过。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# 检测 Ollama 是否可用
_OLLAMA_AVAILABLE = False
try:
    import requests

    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    if resp.status_code == 200:
        models = [m["name"] for m in resp.json().get("models", [])]
        _OLLAMA_AVAILABLE = "bge-m3:latest" in models or any(
            n.startswith("bge-m3") for n in models
        )
except Exception:
    pass

skip_no_ollama = pytest.mark.skipif(
    not _OLLAMA_AVAILABLE,
    reason="Ollama 服务不可用或未安装 nomic-embed-text",
)


# ==================== OllamaEmbedding 测试 ====================


@skip_no_ollama
class TestOllamaEmbedding:
    """测试 Ollama 嵌入管道。"""

    def test_dimension(self):
        """嵌入维度 = 1024（bge-m3）。"""
        from danniao.hippocampus import OllamaEmbedding

        emb = OllamaEmbedding()
        assert emb.dimension == 1024

    def test_embed_text_returns_normalized(self):
        """embed_text 返回归一化向量。"""
        from danniao.hippocampus import OllamaEmbedding

        emb = OllamaEmbedding()
        vec = emb.embed_text("苹果")

        assert vec.shape == (1024,)
        assert vec.dtype == np.float32
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 0.01  # L2 归一化

    def test_embed_batch(self):
        """embed_batch 返回正确形状的矩阵。"""
        from danniao.hippocampus import OllamaEmbedding

        emb = OllamaEmbedding()
        vecs = emb.embed_batch(["苹果", "梨", "香蕉"])

        assert vecs.shape == (3, 1024)
        # 每行归一化
        for i in range(3):
            norm = float(np.linalg.norm(vecs[i]))
            assert abs(norm - 1.0) < 0.01

    def test_same_text_same_vector(self):
        """相同文本返回相同向量（确定性）。"""
        from danniao.hippocampus import OllamaEmbedding

        emb = OllamaEmbedding()
        v1 = emb.embed_text("确定性测试")
        v2 = emb.embed_text("确定性测试")

        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_semantic_similarity_fruits(self):
        """同类概念（水果）相似度高于不同类（水果 vs 交通工具）。"""
        from danniao.hippocampus import OllamaEmbedding

        emb = OllamaEmbedding()
        # 使用完整句子以获得更清晰的语义区分
        vec_apple = emb.embed_text("苹果是一种水果，味道甜美")
        vec_pear = emb.embed_text("梨也是一种水果，多汁可口")
        vec_car = emb.embed_text("汽车是交通工具，在公路上行驶")

        sim_fruit = float(np.dot(vec_apple, vec_pear))
        sim_cross = float(np.dot(vec_apple, vec_car))

        assert sim_fruit > sim_cross, f"水果相似度 {sim_fruit} 应高于跨类 {sim_cross}"

    def test_empty_batch(self):
        """空批量返回空矩阵。"""
        from danniao.hippocampus import OllamaEmbedding

        emb = OllamaEmbedding()
        vecs = emb.embed_batch([])

        assert vecs.shape[0] == 0

    def test_connection_error_on_bad_host(self):
        """错误的主机地址 → RuntimeError。"""
        from danniao.hippocampus import OllamaEmbedding

        with pytest.raises(RuntimeError, match="无法连接"):
            OllamaEmbedding(host="http://localhost:99999")


# ==================== LLMExpressionEngine 测试 ====================


@skip_no_ollama
class TestLLMExpressionEngine:
    """测试 LLM 表达引擎。"""

    def test_llm_generates_text(self):
        """LLM 生成非空文本。"""
        import networkx as nx

        from danniao.expression import ExpressionContext, LLMExpressionEngine

        class _MockTree:
            def __init__(self):
                self.graph = nx.DiGraph()

        engine = LLMExpressionEngine(_MockTree())
        ctx = ExpressionContext(
            input_text="量子力学",
            matched_trunk="量子力学",
            is_new_trunk=True,
            curiosity=0.8,
            energy=0.9,
        )

        result = engine.express(ctx)

        assert result.text != ""
        assert len(result.text) > 0
        # LLM 可用时应使用 LLM 模式
        assert result.mode in ("llm", "template")

    def test_llm_match_expression(self):
        """LLM 对已知概念生成表达。"""
        import networkx as nx

        from danniao.expression import ExpressionContext, LLMExpressionEngine

        class _MockTree:
            def __init__(self):
                self.graph = nx.DiGraph()
                self.graph.add_node("n1", label="苹果", kind="trunk")

        engine = LLMExpressionEngine(_MockTree())
        ctx = ExpressionContext(
            input_text="青苹果",
            matched_trunk="苹果",
            is_new_trunk=False,
            prediction_error=0.1,
            curiosity=0.5,
            energy=0.8,
        )

        result = engine.express(ctx)

        assert result.text != ""
        assert result.mode in ("llm", "template")

    def test_fallback_to_template_on_error(self):
        """LLM 不可用时回退到模板表达。"""
        import networkx as nx

        from danniao.expression import ExpressionContext, LLMExpressionEngine

        class _MockTree:
            def __init__(self):
                self.graph = nx.DiGraph()

        # 使用不存在的模型 → LLM 不可用 → 回退模板
        engine = LLMExpressionEngine(_MockTree(), model_name="nonexistent-model")
        ctx = ExpressionContext(
            input_text="测试",
            matched_trunk="测试",
            is_new_trunk=True,
        )

        result = engine.express(ctx)

        assert result.text != ""
        assert result.mode == "template"  # 回退到模板


# ==================== 端到端集成测试 ====================


@skip_no_ollama
class TestEndToEndWithOllama:
    """端到端测试：真实 Ollama 嵌入 + ChromaDB + 认知树。"""

    def test_full_pipeline_with_ollama(self):
        """完整链路：Ollama 嵌入 → 门控 → 扩散 → 表达。"""
        from danniao.expression import LLMExpressionEngine
        from danniao.hippocampus import (
            ChromaVectorStore,
            DynamicCognitiveTree,
            EpisodicLog,
            InformationTriggerGate,
            NeuroDynamicsEngine,
            OllamaEmbedding,
            SpreadingActivation,
            VectorCognitiveSpace,
        )
        from danniao.mind import ContinuousMind
        from danniao.motivation import (
            ExplorationEngine,
            Homeostasis,
            RewardSystem,
        )

        emb = OllamaEmbedding()
        store = ChromaVectorStore(
            path=".test_chroma_e2e_ollama",
            collection_name="test_e2e_ollama",
            expected_dim=emb.dimension,
        )
        space = VectorCognitiveSpace(vector_store=store, embedding=emb)
        tree = DynamicCognitiveTree(space)
        gate = InformationTriggerGate(tree)
        dynamics = NeuroDynamicsEngine(tree.graph)
        spreading = SpreadingActivation(space)
        homeostasis = Homeostasis()
        log = EpisodicLog(path=":memory:")

        expression_engine = LLMExpressionEngine(tree)
        exploration_engine = ExplorationEngine(tree, homeostasis)
        reward_system = RewardSystem(homeostasis, dynamics)

        mind = ContinuousMind(
            tree=tree,
            gate=gate,
            dynamics=dynamics,
            spreading=spreading,
            homeostasis=homeostasis,
            episodic_log=log,
            expression_engine=expression_engine,
            exploration_engine=exploration_engine,
            reward_system=reward_system,
        )

        # 1. 第一次输入：新概念
        r1 = mind.process("苹果")
        assert r1.is_new_trunk is True
        assert r1.expression != ""

        # 2. 第二次输入：相同概念
        r2 = mind.process("苹果")
        assert r2.is_new_trunk is False
        assert r2.expression != ""

        # 3. 输入相似概念
        r3 = mind.process("梨")
        assert r3.expression != ""

        # 4. 给反馈
        feedback = mind.give_feedback(success=True)
        assert feedback is not None
        assert feedback.success is True

    def test_semantic_matching_with_ollama(self):
        """bge-m3 语义匹配：青苹果匹配苹果（同类，相似度 > 阈值）。"""
        from danniao.hippocampus import (
            ChromaVectorStore,
            DynamicCognitiveTree,
            InformationTriggerGate,
            OllamaEmbedding,
            VectorCognitiveSpace,
        )

        emb = OllamaEmbedding()
        store = ChromaVectorStore(
            path=".test_chroma_semantic_bge",
            collection_name="test_semantic_bge",
            expected_dim=emb.dimension,
        )
        space = VectorCognitiveSpace(vector_store=store, embedding=emb)
        tree = DynamicCognitiveTree(space)
        gate = InformationTriggerGate(tree)

        # 创建苹果主干
        gate.process("苹果")

        # 青苹果应该匹配苹果（同类水果，bge-m3 相似度 > 0.65）
        result = gate.process("青苹果")
        assert result.matched_trunk == "苹果"
        assert result.matched_trunk_similarity > 0.65
