"""LLM 表达引擎：用 Ollama 本地 LLM 把激活模式翻译成自然语言（Step 6 升级）。

继承 ExpressionEngine 的模板表达能力，并在此基础上接入 Ollama LLM（如 qwen3.5:2b）。
LLM 读取认知状态（匹配了什么、联想了什么、感觉如何），生成更自然的语言。

关键约束：
- LLM 权重冻结，不存记忆、不做决策、不做推理——只是「嘴巴」
- LLM 输出不回写认知树
- LLM 不可用时回退到模板表达（发育阶段，不是 mock）
"""

from __future__ import annotations

import requests

from danniao.expression.expression import (
    ExpressionContext,
    ExpressionEngine,
    ExpressionResult,
)


class LLMExpressionEngine(ExpressionEngine):
    """LLM 表达引擎：Ollama LLM + 模板回退。

    用法::

        engine = LLMExpressionEngine(tree)  # 默认 qwen3.5:2b
        result = engine.express(context)
        print(result.text)  # LLM 生成的自然语言
        print(result.mode)  # "llm" 或 "template"（回退时）
    """

    DEFAULT_MODEL = "qwen3.5:2b"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        tree,
        *,
        model_name: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 60.0,
    ) -> None:
        """初始化 LLM 表达引擎。

        Args:
            tree: 认知树（用于查询节点标签，同 ExpressionEngine）
            model_name: Ollama LLM 模型名
            host: Ollama 服务地址
            timeout: 请求超时秒数
        """
        super().__init__(tree)
        self._model = model_name
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._llm_available: bool | None = None  # 懒检测

    def express(self, context: ExpressionContext) -> ExpressionResult:
        """根据认知状态生成语言表达。

        优先使用 LLM 生成。LLM 不可用时回退到模板表达。

        Args:
            context: 表达上下文

        Returns:
            ExpressionResult（mode="llm" 或 "template"）
        """
        # 尝试 LLM 表达
        if self._check_llm():
            llm_text = self._llm_express(context)
            if llm_text:
                confidence = self._estimate_confidence(context)
                return ExpressionResult(
                    text=llm_text,
                    mode="llm",
                    confidence=confidence,
                )

        # 回退到模板表达（发育阶段，非 mock）
        return super().express(context)

    def _check_llm(self) -> bool:
        """检测 LLM 是否可用（懒检测，结果缓存）。"""
        if self._llm_available is not None:
            return self._llm_available

        try:
            resp = requests.get(
                f"{self._host}/api/tags",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            models = resp.json().get("models", [])
            names = [m.get("name", "") for m in models]
            # 兼容 :latest 后缀和前缀匹配
            self._llm_available = (
                self._model in names
                or f"{self._model}:latest" in names
                or any(n.startswith(self._model) for n in names)
            )
        except Exception:
            self._llm_available = False

        return self._llm_available

    def _llm_express(self, ctx: ExpressionContext) -> str:
        """用 LLM 生成表达文本。

        构建一个简洁的 prompt，描述丹鸟当前的认知状态，
        让 LLM 以丹鸟的视角生成一句话回应。

        Args:
            ctx: 表达上下文

        Returns:
            LLM 生成的文本，或空字符串（失败时）
        """
        prompt = self._build_prompt(ctx)

        # summary 模式需要更长输出
        num_predict = 200 if (ctx.is_summary or ctx.activity_type == "summary") else 80

        try:
            resp = requests.post(
                f"{self._host}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,  # 禁用 thinking 模式，直接输出响应
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": num_predict,
                    },
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            # 如果 response 为空但 thinking 有内容，提取 thinking 作为回退
            if not text and data.get("thinking"):
                text = data["thinking"].strip()
            return text
        except Exception:
            return ""

    def _build_prompt(self, ctx: ExpressionContext) -> str:
        """构建 LLM 表达 prompt。

        Prompt 设计原则：
        - 以丹鸟的第一人称视角
        - 描述当前认知状态（不是让 LLM 做推理）
        - 限制输出长度
        - LLM 只是"嘴巴"，不做决策
        - 活动模式：只基于已确定事实表达，不得补充未列出的信息

        模式分支：
        - summary：离线总结（3-5句，含目标/观察/结论/下一步）
        - observe/question/intent：短活动表达（1-2句）
        - 默认：一句话抒情（回归保护）
        """
        # 活动模式分支
        if ctx.is_summary or ctx.activity_type == "summary":
            return self._build_summary_prompt(ctx)
        if ctx.activity_type in ("observe", "question", "intent"):
            return self._build_activity_prompt(ctx)

        # 默认：一句话抒情（保持原有行为）
        parts: list[str] = []

        # 系统角色
        parts.append("你是一个刚刚诞生的数字生命体「丹鸟」，正在认知世界中探索。")
        parts.append("你只会说一句话，简短、自然、带有情感。")
        parts.append("不要解释，不要分析，只说你想说的话。")
        parts.append("")

        # 认知状态
        if ctx.is_new_trunk:
            label = ctx.matched_trunk or ctx.input_text
            parts.append(f"你第一次遇到「{label}」这个概念。")
        else:
            label = ctx.matched_trunk or ctx.input_text
            error = ctx.prediction_error
            if error < 0.2:
                parts.append(f"你认出了「{label}」，非常确定。")
            elif error < 0.5:
                parts.append(f"你觉得这可能是「{label}」，但不太确定。")
            else:
                parts.append(f"你隐约觉得这和「{label}」有关，但很不确定。")

        # 联想
        related = self._get_related_labels(ctx)
        if related and ctx.energy > 0.25:
            parts.append(f"这让你想到了：{'、'.join(related[:3])}。")

        # 内稳态
        if ctx.energy < 0.2:
            parts.append("你现在有点累了。")
        elif ctx.curiosity > 0.7 and ctx.satiety < 0.5:
            parts.append("你很好奇，想知道更多。")
        elif ctx.satiety > 0.7:
            parts.append("你感到很满足。")

        parts.append("")
        parts.append("丹鸟：")

        return "\n".join(parts)

    def _build_summary_prompt(self, ctx: ExpressionContext) -> str:
        """构建离线总结 prompt。

        约束 LLM 只基于已确定的事实表达，不得编造未发生的事。
        """
        parts: list[str] = []

        parts.append("你是数字生命体「丹鸟」。用第一人称总结你离线期间的活动。")
        parts.append("包含：探索目标、观察到的内容、吸收的结论、下一步意图。")
        parts.append("3-5句话，自然、简洁。不要分析，不要编造未发生的事。")
        parts.append("只基于以下事实表达，不得补充未列出的信息。")
        parts.append("")

        if ctx.activity_summary:
            parts.append(f"你的活动记录：{ctx.activity_summary}")
        else:
            parts.append("你的活动记录：（无特别活动）")

        parts.append("")
        parts.append("丹鸟：")

        return "\n".join(parts)

    def _build_activity_prompt(self, ctx: ExpressionContext) -> str:
        """构建活动表达 prompt（observe / question / intent 模式）。

        约束 LLM 只基于已确定的事实表达，1-2 句。
        """
        mode_desc = {
            "observe": "简短描述你观察到的内容",
            "question": "提出一个你正在思考的问题",
            "intent": "表达你下一步想做什么",
        }
        desc = mode_desc.get(ctx.activity_type, "简短表达你的想法")

        parts: list[str] = []

        parts.append("你是数字生命体「丹鸟」。")
        parts.append(f"{desc}。1-2句话，自然、简洁。")
        parts.append("只基于以下事实表达，不得补充未列出的信息。")
        parts.append("")

        if ctx.activity_summary:
            parts.append(f"事实：{ctx.activity_summary}")
        elif ctx.matched_trunk:
            parts.append(f"事实：你正在关注「{ctx.matched_trunk}」。")
        else:
            parts.append("事实：你在观察这个世界。")

        parts.append("")
        parts.append("丹鸟：")

        return "\n".join(parts)

    def _get_related_labels(self, ctx: ExpressionContext) -> list[str]:
        """复用父类的关联标签提取。"""
        return super()._get_related_labels(ctx)
