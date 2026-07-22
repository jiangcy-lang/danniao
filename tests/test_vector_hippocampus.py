"""向量海马体验收：embedding 双写 + 余弦匹配门控。"""

from __future__ import annotations

from danniao.hippocampus import DynamicCognitiveTree, InformationTriggerGate
from danniao.hippocampus.embeddings import HashTextEmbedder
from danniao.hippocampus.vector_store import HippocampusVectorStore


def _make_tree() -> DynamicCognitiveTree:
    return DynamicCognitiveTree(
        embedder=HashTextEmbedder(),
        vector_store=HippocampusVectorStore(ephemeral=True),
    )


def test_nodes_have_embeddings() -> None:
    tree = _make_tree()
    try:
        tree.add_trunk("苹果")
        assert tree.has_embedding("苹果")
        tree.add_feature_child("苹果", "颜色", "红")
        assert tree.has_embedding("颜色-红")
        assert tree.vector_store.count() == 2
    finally:
        tree.close()


def test_cosine_match_routine_activate() -> None:
    tree = _make_tree()
    try:
        gate = InformationTriggerGate(tree, similarity_threshold=0.85)
        gate.process("苹果")
        r2 = gate.process("苹果")
        assert r2.action == "routine_activate"
        assert r2.cosine_similarity is not None
        assert r2.cosine_similarity >= 0.85
        assert tree.feature_count("苹果") == 0
    finally:
        tree.close()


def test_high_info_spawns_with_vector_match() -> None:
    tree = _make_tree()
    try:
        gate = InformationTriggerGate(tree)
        gate.process("苹果")
        r = gate.process("红色的甜苹果")
        assert r.action == "spawned_children"
        assert r.cosine_similarity is not None and r.cosine_similarity >= 0.85
        children = {c["concept"] for c in tree.get_children("苹果")}
        assert children == {"颜色-红", "味道-甜"}
    finally:
        tree.close()
