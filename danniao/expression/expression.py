"""表达引擎核心：把认知激活模式翻译成自然语言（Step 6B）。

表达引擎是丹鸟的「嘴巴」。它读取当前的认知状态——
匹配了什么主干、扩散激活了哪些关联节点、内稳态驱动力如何——
然后生成语言输出。

关键约束：
- 只读认知状态，不回写认知树（LLM 是嘴巴不是大脑）
- 模板表达是发育的合法阶段（婴儿蹦单词），不是兜底
- 未来 Step 7+ 可接入小型 LM 实现更自然的表达
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from danniao.hippocampus.tree import DynamicCognitiveTree


@dataclass
class ExpressionContext:
    """表达上下文 —— 从认知状态中提取的表达输入。

    与 ProcessResult 解耦，避免循环依赖。
    ContinuousMind 负责从 ProcessResult + Homeostasis 构建此上下文。
    """

    input_text: str
    """输入文本。"""

    matched_trunk: str | None = None
    """匹配到的主干标签（新主干时为输入文本本身）。"""

    is_new_trunk: bool = False
    """是否是新概念（首次遇到）。"""

    prediction_error: float = 0.0
    """预测误差（0~1）。低 = 高置信度匹配。"""

    activated_nodes: dict[str, float] = field(default_factory=dict)
    """扩散激活的节点 {node_id: activation_level}。"""

    curiosity: float = 0.5
    """好奇心（0~1）。"""

    confidence: float = 0.3
    """置信度（0~1）。"""

    energy: float = 1.0
    """能量（0~1）。低 = 疲倦，表达变简短。"""

    satiety: float = 0.2
    """满足感（0~1）。高 = 满足，表达更从容。"""

    activity_type: str = ""
    """活动类型：observe / question / intent / summary / ""（空表示非活动表达）。"""

    activity_summary: str = ""
    """活动结果摘要（供 LLM 或模板作为事实输入）。"""

    is_bubble: bool = False
    """是否为主动冒泡。"""

    is_summary: bool = False
    """是否为离线总结。"""


@dataclass
class ExpressionResult:
    """表达结果。"""

    text: str
    """生成的语言文本。"""

    mode: str = "template"
    """表达模式：当前为 "template"，未来可扩展 "llm"。"""

    confidence: float = 0.0
    """表达置信度（0~1）。反映丹鸟对这次输出的把握程度。"""


class ExpressionEngine:
    """表达引擎：把激活模式翻译成语言。

    三层表达结构：
    1. 核心：识别了什么？（新概念 / 已知概念匹配）
    2. 联想：想到了什么？（扩散激活的关联概念）
    3. 驱动：感觉如何？（内稳态驱动的情感色彩）

    用法::

        engine = ExpressionEngine(tree)
        context = ExpressionContext(
            input_text="青苹果",
            matched_trunk="苹果",
            is_new_trunk=False,
            prediction_error=0.12,
            activated_nodes={...},
            curiosity=0.67,
        )
        result = engine.express(context)
        print(result.text)  # "我知道！这是「苹果」。让我想到……"
    """

    def __init__(self, tree: DynamicCognitiveTree) -> None:
        """初始化表达引擎。

        Args:
            tree: 认知树（用于查询节点标签）
        """
        self.tree = tree

    def express(self, context: ExpressionContext) -> ExpressionResult:
        """根据认知状态生成语言表达。

        Args:
            context: 表达上下文

        Returns:
            ExpressionResult 表达结果
        """
        # 活动表达：冒泡/总结/观察等模式
        if context.is_bubble or context.is_summary or context.activity_type:
            return self._express_activity(context)

        parts: list[str] = []

        # 1. 核心表达：新概念 or 已知匹配
        core = self._express_core(context)
        parts.append(core)

        # 2. 联想表达：扩散激活的关联概念
        # 能量太低时跳过联想（疲倦时表达变简短）
        if context.energy > 0.25:
            assoc = self._express_association(context)
            if assoc:
                parts.append(assoc)

        # 3. 驱动表达：内稳态的情感色彩
        drive = self._express_drive(context)
        if drive:
            parts.append(drive)

        text = "".join(parts)
        confidence = self._estimate_confidence(context)

        return ExpressionResult(
            text=text,
            mode="template",
            confidence=confidence,
        )

    # ==================== 活动表达 ====================

    def _express_activity(self, ctx: ExpressionContext) -> ExpressionResult:
        """活动表达：观察/疑问/意图/总结可切换模式。

        当上下文标记为冒泡、总结或活动类型时走此分支。
        模板阶段产出结构化短文，不是单句抒情。
        """
        if ctx.is_summary or ctx.activity_type == "summary":
            return self._express_summary(ctx)
        if ctx.activity_type == "observe":
            return self._express_observe(ctx)
        if ctx.activity_type == "question":
            return self._express_question(ctx)
        if ctx.activity_type == "intent":
            return self._express_intent(ctx)
        # 默认冒泡：简短观察
        return self._express_observe(ctx)

    def _express_summary(self, ctx: ExpressionContext) -> ExpressionResult:
        """离线总结：目标/观察/结论/下一步四段结构。"""
        parts: list[str] = []

        if ctx.activity_summary:
            parts.append(ctx.activity_summary)
        else:
            if ctx.matched_trunk:
                parts.append(f"我探索了「{ctx.matched_trunk}」。")
            parts.append("我尝试寻找，但证据还不够。")

        text = "".join(parts) if len(parts) == 1 else "\n".join(parts)
        return ExpressionResult(
            text=text,
            mode="template_activity",
            confidence=0.5,
        )

    def _express_observe(self, ctx: ExpressionContext) -> ExpressionResult:
        """观察型冒泡：简短描述当前认知状态。"""
        if ctx.activity_summary:
            return ExpressionResult(
                text=ctx.activity_summary,
                mode="template_activity",
                confidence=0.4,
            )
        if ctx.matched_trunk:
            return ExpressionResult(
                text=f"我注意到「{ctx.matched_trunk}」。",
                mode="template_activity",
                confidence=0.4,
            )
        return ExpressionResult(
            text="我在观察这个世界。",
            mode="template_activity",
            confidence=0.3,
        )

    def _express_question(self, ctx: ExpressionContext) -> ExpressionResult:
        """疑问型冒泡：提出一个问题。"""
        if ctx.activity_summary:
            return ExpressionResult(
                text=ctx.activity_summary,
                mode="template_activity",
                confidence=0.4,
            )
        if ctx.matched_trunk:
            return ExpressionResult(
                text=f"「{ctx.matched_trunk}」到底是什么呢？",
                mode="template_activity",
                confidence=0.3,
            )
        return ExpressionResult(
            text="我在想一些事情。",
            mode="template_activity",
            confidence=0.3,
        )

    def _express_intent(self, ctx: ExpressionContext) -> ExpressionResult:
        """意图型冒泡：表达下一步想做什么。"""
        if ctx.activity_summary:
            return ExpressionResult(
                text=ctx.activity_summary,
                mode="template_activity",
                confidence=0.4,
            )
        if ctx.matched_trunk:
            return ExpressionResult(
                text=f"我想继续了解「{ctx.matched_trunk}」。",
                mode="template_activity",
                confidence=0.4,
            )
        return ExpressionResult(
            text="我想探索更多。",
            mode="template_activity",
            confidence=0.3,
        )

    # ==================== 核心表达 ====================

    def _express_core(self, ctx: ExpressionContext) -> str:
        """核心表达：识别了什么。"""
        if ctx.is_new_trunk:
            return self._express_new(ctx)
        return self._express_match(ctx)

    def _express_new(self, ctx: ExpressionContext) -> str:
        """新概念表达：惊讶 + 好奇。"""
        label = ctx.matched_trunk or ctx.input_text

        if ctx.curiosity > 0.6 and ctx.energy > 0.4:
            return f"「{label}」……这是什么？我从没见过。"
        if ctx.energy <= 0.3:
            return f"……{label}。新东西。"
        return f"「{label}」……新东西。"

    def _express_match(self, ctx: ExpressionContext) -> str:
        """已知概念匹配表达：置信度调制。"""
        label = ctx.matched_trunk or ctx.input_text
        error = ctx.prediction_error

        if error < 0.2:
            # 高置信度匹配
            if ctx.satiety > 0.6:
                return f"嗯，「{label}」，我很熟悉。"
            return f"我知道！这是「{label}」。"

        if error < 0.5:
            # 中等置信度
            return f"这是「{label}」吧。"

        # 低置信度
        return f"嗯……好像是「{label}」？"

    # ==================== 联想表达 ====================

    def _express_association(self, ctx: ExpressionContext) -> str:
        """联想表达：扩散激活的关联概念。"""
        related = self._get_related_labels(ctx)

        if not related:
            return ""

        if len(related) == 1:
            return f"让我想到{related[0]}。"
        if len(related) == 2:
            return f"让我想到{related[0]}和{related[1]}。"
        # 3+ 个关联
        labels = "、".join(related[:3])
        return f"让我想到{labels}……"

    def _get_related_labels(
        self,
        ctx: ExpressionContext,
        *,
        max_count: int = 3,
        min_activation: float = 0.1,
    ) -> list[str]:
        """从扩散激活结果中提取关联概念标签。

        排除主干节点本身，按激活水平降序排列。

        Args:
            ctx: 表达上下文
            max_count: 最多返回的关联概念数
            min_activation: 最低激活水平阈值

        Returns:
            标签列表
        """
        if not ctx.activated_nodes:
            return []

        # 按激活水平降序排列
        sorted_nodes = sorted(
            ctx.activated_nodes.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # 找到主干节点 ID 以排除
        trunk_id = None
        if ctx.matched_trunk:
            for nid, nd in self.tree.graph.nodes(data=True):
                if nd.get("label") == ctx.matched_trunk:
                    trunk_id = nid
                    break

        labels: list[str] = []
        seen: set[str] = set()
        for nid, level in sorted_nodes:
            if level < min_activation:
                break
            if nid == trunk_id:
                continue
            if nid in seen:
                continue

            label = self._node_label(nid)
            if label and label not in seen:
                labels.append(label)
                seen.add(label)

            if len(labels) >= max_count:
                break

        return labels

    # ==================== 驱动表达 ====================

    def _express_drive(self, ctx: ExpressionContext) -> str:
        """驱动表达：内稳态的情感色彩。

        只在驱动力足够强时才附加，避免表达过于冗长。
        """
        # 能量极低 → 表达疲倦
        if ctx.energy < 0.2:
            return "……有点累了。"

        # 好奇心极高 → 表达探索欲
        if ctx.curiosity > 0.7 and ctx.satiety < 0.5:
            return "我想知道更多。"

        # 满足感极高 → 表达满足
        if ctx.satiety > 0.7:
            return "嗯，很满足。"

        return ""

    # ==================== 工具方法 ====================

    def _node_label(self, node_id: str) -> str:
        """获取节点的标签。"""
        if node_id in self.tree.graph:
            return str(
                self.tree.graph.nodes[node_id].get("label", node_id)
            )
        return ""

    def _estimate_confidence(self, ctx: ExpressionContext) -> float:
        """估计表达置信度。

        新概念 = 0（完全不确定）。
        匹配概念 = 1 - prediction_error（匹配越好越自信）。
        """
        if ctx.is_new_trunk:
            return 0.0
        return max(0.0, 1.0 - ctx.prediction_error)
