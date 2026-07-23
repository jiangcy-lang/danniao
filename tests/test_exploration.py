"""Step 6C 验收测试：探索引擎。

验证好奇心驱动的主动探索目标生成。
所有测试无需外部依赖。
"""

from __future__ import annotations

import networkx as nx

from danniao.motivation import (
    Drive,
    ExplorationEngine,
    ExplorationTarget,
    Homeostasis,
    HomeostasisConfig,
)


# ==================== 工具 ====================


class _MockTree:
    """最小认知树桩，仅供探索引擎测试。"""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_trunk(self, node_id: str, label: str) -> None:
        self.graph.add_node(
            node_id, label=label, kind="trunk", activation_weight=0.5
        )

    def add_feature(self, node_id: str, label: str, parent_id: str) -> None:
        self.graph.add_node(
            node_id, label=label, kind="feature", activation_weight=0.3
        )
        self.graph.add_edge(parent_id, node_id, weight=0.2)

    def connect(self, source: str, target: str) -> None:
        self.graph.add_edge(source, target, weight=0.3)

    def trunk_count(self) -> int:
        return sum(
            1
            for _, d in self.graph.nodes(data=True)
            if d.get("kind") == "trunk"
        )


def _curious_homeostasis() -> Homeostasis:
    """创建好奇心高的内稳态（想探索状态）。"""
    homeo = Homeostasis()
    homeo.state.curiosity = 0.8
    homeo.state.energy = 0.9
    homeo.state.satiety = 0.2
    return homeo


def _lethargic_homeostasis() -> Homeostasis:
    """创建不想探索的内稳态（低能量）。"""
    homeo = Homeostasis()
    homeo.state.curiosity = 0.3
    homeo.state.energy = 0.1
    homeo.state.satiety = 0.8
    return homeo


def _satiated_homeostasis() -> Homeostasis:
    """创建不想探索的内稳态（高满足）。"""
    homeo = Homeostasis()
    homeo.state.curiosity = 0.6
    homeo.state.energy = 0.8
    homeo.state.satiety = 0.8
    return homeo


# ==================== 不想探索时 ====================


def test_no_exploration_when_lethargic():
    """低能量 → 不探索。"""
    engine = ExplorationEngine(_MockTree(), _lethargic_homeostasis())
    assert engine.propose([]) is None


def test_no_exploration_when_satiated():
    """高满足 → 不探索。"""
    engine = ExplorationEngine(_MockTree(), _satiated_homeostasis())
    assert engine.propose([]) is None


# ==================== 求新探索 ====================


def test_novelty_when_tree_empty():
    """认知树空 + 想探索 → 求新探索。"""
    tree = _MockTree()
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose([])

    assert target is not None
    assert target.exploration_type == "novelty"
    assert "新东西" in target.text or "新事物" in target.text
    assert target.target_node_ids == []


# ==================== 深度探索 ====================


def test_depth_exploration_few_features():
    """主干特征少 + 想探索 → 深度探索。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose(["n1"])

    assert target is not None
    assert target.exploration_type == "depth"
    assert "苹果" in target.text
    assert "n1" in target.target_node_ids


def test_depth_exploration_no_features():
    """主干无特征 + 想探索 → 深度探索。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose(["n1"])

    assert target is not None
    assert target.exploration_type == "depth"
    assert "了解更多" in target.text


def test_no_depth_when_enough_features():
    """主干有足够特征 → 跳过深度探索，但回退到外部探索。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    tree.add_feature("f1", "颜色-红", "n1")
    tree.add_feature("f2", "味道-甜", "n1")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    # 有足够特征 + 单主干 → 无关联探索、无深度探索、无求新
    target = engine.propose(["n1"])

    # 单主干 + 足够特征 → 回退到外部探索（ask_external）
    assert target is not None
    assert target.exploration_type == "ask_external"
    assert target.needs_external is True


# ==================== 关联探索 ====================


def test_relationship_two_trunks_no_edge():
    """两个近期主干无边 → 关联探索。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    tree.add_trunk("n2", "梨")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose(["n1", "n2"])

    assert target is not None
    assert target.exploration_type == "relationship"
    assert "苹果" in target.text
    assert "梨" in target.text
    assert "关系" in target.text
    assert "n1" in target.target_node_ids
    assert "n2" in target.target_node_ids


def test_no_relationship_when_already_connected():
    """两个主干已有边 → 不做关联探索（已有关系）。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    tree.add_trunk("n2", "梨")
    tree.connect("n1", "n2")  # 已有关联
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose(["n1", "n2"])

    # 已有关联 → 尝试深度探索（两个主干特征都少）
    assert target is not None
    assert target.exploration_type == "depth"


def test_relationship_priority_over_depth():
    """关联探索优先于深度探索。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    tree.add_trunk("n2", "梨")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose(["n1", "n2"])

    # 两个主干都缺特征，但关联探索优先
    assert target is not None
    assert target.exploration_type == "relationship"


# ==================== 近期节点过滤 ====================


def test_recent_feature_nodes_ignored():
    """近期节点中的特征节点被忽略，只关注主干。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    tree.add_feature("f1", "颜色-红", "n1")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    # 只传特征节点 → 找不到主干 → 深度探索失败 → novelty（trunk_count > 0 → None）
    target = engine.propose(["f1"])

    assert target is None


def test_recent_unknown_nodes_ignored():
    """不在图中的节点 ID 被忽略。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose(["unknown_id", "n1"])

    assert target is not None
    assert target.exploration_type == "depth"
    assert "苹果" in target.text


def test_empty_recent_nodes():
    """空近期节点 + 有主干 → 深度探索失败 → novelty 失败 → None。"""
    tree = _MockTree()
    tree.add_trunk("n1", "苹果")
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose([])

    # 有主干但无近期 → 深度探索找不到目标 → novelty 需要 trunk_count==0 → None
    assert target is None


# ==================== 紧迫度和驱动力 ====================


def test_urgency_reflects_curiosity():
    """紧迫度反映好奇心水平。"""
    tree = _MockTree()
    homeo = _curious_homeostasis()
    homeo.state.curiosity = 0.9
    engine = ExplorationEngine(tree, homeo)

    target = engine.propose([])

    assert target is not None
    assert target.urgency >= 0.8


def test_drive_is_curiosity():
    """探索目标的驱动力是好奇心。"""
    tree = _MockTree()
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose([])

    assert target is not None
    assert target.drive == Drive.CURIOSITY


# ==================== 结果完整性 ====================


def test_target_has_text():
    """探索目标包含文本。"""
    tree = _MockTree()
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose([])

    assert target is not None
    assert len(target.text) > 0


def test_target_has_type():
    """探索目标包含类型。"""
    tree = _MockTree()
    engine = ExplorationEngine(tree, _curious_homeostasis())

    target = engine.propose([])

    assert target is not None
    assert target.exploration_type in ("depth", "relationship", "novelty")
