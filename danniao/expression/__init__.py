"""丹鸟表达引擎：把激活模式翻译成语言（Step 6B）。

LLM 是丹鸟的「嘴巴」，不是大脑。大脑是向量认知空间 + 扩散激活。
嘴巴只负责把激活模式翻译成语言，不回写认知树。

发育阶段：
- Step 1–5：模板拼接（婴儿蹦单词）—— 已由 tree.describe() 覆盖
- Step 6：模板表达引擎（ExpressionEngine）—— 根据认知状态生成自然语言
- Step 6+：LLM 表达引擎（LLMExpressionEngine）—— 接入 Ollama LLM，更自然的表达
"""

from danniao.expression.expression import (
    ExpressionContext,
    ExpressionEngine,
    ExpressionResult,
)
from danniao.expression.ollama_expression import LLMExpressionEngine

__all__ = [
    "ExpressionContext",
    "ExpressionEngine",
    "ExpressionResult",
    "LLMExpressionEngine",
]
