"""Step 6B 验收测试：表达引擎。

验证激活模式 → 自然语言翻译。
所有测试无需外部依赖。
"""

from __future__ import annotations

import networkx as nx

from danniao.expression import ExpressionContext, ExpressionEngine, ExpressionResult


# ==================== 工具 ====================


class _MockTree:
    """最小认知树桩，仅供表达引擎测试。"""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, label: str, kind: str = "trunk") -> None:
        self.graph.add_node(node_id, label=label, kind=kind, activation_weight=0.5)

    def add_edge(self, source: str, target: str, weight: float = 0.3) -> None:
        self.graph.add_edge(source, target, weight=weight)


def _ctx(**kwargs) -> ExpressionContext:
    """快速构建表达上下文。"""
    defaults = dict(input_text="测试")
    defaults.update(kwargs)
    return ExpressionContext(**defaults)


# ==================== 新概念表达 ====================


def test_new_trunk_high_curiosity():
    """新概念 + 高好奇心 + 高能量 → 惊讶好奇的表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="量子纠缠",
        matched_trunk="量子纠缠",
        is_new_trunk=True,
        curiosity=0.8,
        energy=0.9,
    )

    result = engine.express(ctx)

    assert "量子纠缠" in result.text
    assert "什么" in result.text or "没见过" in result.text
    assert result.confidence == 0.0


def test_new_trunk_low_energy():
    """新概念 + 低能量 → 简短疲倦的表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="量子纠缠",
        matched_trunk="量子纠缠",
        is_new_trunk=True,
        curiosity=0.3,
        energy=0.15,
    )

    result = engine.express(ctx)

    assert "量子纠缠" in result.text
    assert "累" in result.text  # 疲倦表达
    # 低能量不应有联想
    assert "让我想到" not in result.text


def test_new_trunk_normal():
    """新概念 + 正常状态 → 简洁新概念表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="新概念",
        matched_trunk="新概念",
        is_new_trunk=True,
        curiosity=0.4,
        energy=0.8,
    )

    result = engine.express(ctx)

    assert "新概念" in result.text
    assert "新东西" in result.text


# ==================== 已知匹配表达 ====================


def test_match_high_confidence():
    """高置信度匹配（低误差）→ 自信表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="青苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.08,
        confidence=0.9,
    )

    result = engine.express(ctx)

    assert "苹果" in result.text
    assert "我知道" in result.text or "这是" in result.text
    assert result.confidence > 0.8


def test_match_medium_confidence():
    """中等置信度匹配 → 犹豫表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="梨子",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.35,
    )

    result = engine.express(ctx)

    assert "苹果" in result.text
    assert "吧" in result.text  # 犹豫语气


def test_match_low_confidence():
    """低置信度匹配（高误差）→ 不确定表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="汽车",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.6,
    )

    result = engine.express(ctx)

    assert "苹果" in result.text
    assert "好像" in result.text or "？" in result.text


def test_match_high_satiety():
    """高满足感 + 高匹配 → 从容熟悉表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="青苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.05,
        satiety=0.8,
    )

    result = engine.express(ctx)

    assert "苹果" in result.text
    assert "熟悉" in result.text or "满足" in result.text


# ==================== 联想表达 ====================


def test_association_single():
    """单个关联概念 → 联想表达。"""
    tree = _MockTree()
    tree.add_node("n1", "苹果")
    tree.add_node("n2", "红色")

    engine = ExpressionEngine(tree)
    ctx = _ctx(
        input_text="青苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        activated_nodes={"n1": 1.0, "n2": 0.4},
    )

    result = engine.express(ctx)

    assert "红色" in result.text
    assert "让我想到" in result.text


def test_association_multiple():
    """多个关联概念 → 列举联想。"""
    tree = _MockTree()
    tree.add_node("n1", "苹果")
    tree.add_node("n2", "红色")
    tree.add_node("n3", "甜味")
    tree.add_node("n4", "圆形")

    engine = ExpressionEngine(tree)
    ctx = _ctx(
        input_text="青苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        activated_nodes={
            "n1": 1.0,
            "n2": 0.5,
            "n3": 0.4,
            "n4": 0.3,
        },
    )

    result = engine.express(ctx)

    assert "红色" in result.text
    assert "甜味" in result.text
    assert "让我想到" in result.text


def test_association_excludes_trunk():
    """联想表达排除主干自身。"""
    tree = _MockTree()
    tree.add_node("trunk1", "苹果")
    tree.add_node("n2", "红色")

    engine = ExpressionEngine(tree)
    ctx = _ctx(
        input_text="青苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        activated_nodes={"trunk1": 1.0, "n2": 0.5},
    )

    result = engine.express(ctx)

    # 主干"苹果"已在核心表达中，联想部分不应重复
    assert result.text.count("苹果") == 1
    assert "红色" in result.text


def test_no_association_when_empty():
    """无扩散激活 → 无联想表达。"""
    tree = _MockTree()
    tree.add_node("n1", "苹果")

    engine = ExpressionEngine(tree)
    ctx = _ctx(
        input_text="青苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        activated_nodes={"n1": 1.0},
    )

    result = engine.express(ctx)

    assert "让我想到" not in result.text


def test_low_energy_skips_association():
    """低能量 → 跳过联想表达。"""
    tree = _MockTree()
    tree.add_node("n1", "苹果")
    tree.add_node("n2", "红色")

    engine = ExpressionEngine(tree)
    ctx = _ctx(
        input_text="青苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        energy=0.2,
        activated_nodes={"n1": 1.0, "n2": 0.5},
    )

    result = engine.express(ctx)

    assert "让我想到" not in result.text


# ==================== 驱动表达 ====================


def test_drive_tired():
    """极低能量 → 疲倦表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        energy=0.1,
    )

    result = engine.express(ctx)

    assert "累" in result.text


def test_drive_curious():
    """高好奇心 + 低满足 → 探索欲表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        curiosity=0.8,
        satiety=0.2,
    )

    result = engine.express(ctx)

    assert "想知道更多" in result.text


def test_drive_satisfied():
    """高满足感 → 满足表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.1,
        satiety=0.8,
        curiosity=0.3,
    )

    result = engine.express(ctx)

    assert "满足" in result.text or "熟悉" in result.text


def test_no_drive_expression_when_neutral():
    """中性状态 → 无驱动表达。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        input_text="苹果",
        matched_trunk="苹果",
        is_new_trunk=False,
        prediction_error=0.3,
        curiosity=0.4,
        energy=0.6,
        satiety=0.3,
    )

    result = engine.express(ctx)

    # 中性状态不应有额外的驱动表达
    assert "累" not in result.text
    assert "想知道更多" not in result.text
    assert "满足" not in result.text


# ==================== 置信度 ====================


def test_confidence_new_trunk_zero():
    """新概念 → 置信度 0。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(is_new_trunk=True, matched_trunk="新")

    result = engine.express(ctx)

    assert result.confidence == 0.0


def test_confidence_match_reflects_error():
    """匹配概念 → 置信度 = 1 - 误差。"""
    engine = ExpressionEngine(_MockTree())
    ctx = _ctx(
        is_new_trunk=False,
        matched_trunk="苹果",
        prediction_error=0.15,
    )

    result = engine.express(ctx)

    assert abs(result.confidence - 0.85) < 0.01


# ==================== 结果完整性 ====================


def test_result_has_mode():
    """结果包含表达模式。"""
    engine = ExpressionEngine(_MockTree())
    result = engine.express(_ctx())

    assert result.mode == "template"


def test_result_text_not_empty():
    """结果文本不为空。"""
    engine = ExpressionEngine(_MockTree())
    result = engine.express(_ctx(input_text="测试", matched_trunk="测试", is_new_trunk=True))

    assert len(result.text) > 0
