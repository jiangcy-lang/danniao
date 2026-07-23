"""概念提取器：把输入文本提炼为核心概念（丹鸟的感知理解）。

人脑听到"这个世界有很多好玩的，你对什么感兴趣？"时，
不会把整句话存为一个概念，而是提取出"好奇"、"兴趣"等核心概念。
概念提取器做同样的事——把原始输入转换为认知概念。

短输入（≤8字）直接用作概念，不调用 LLM。
长输入用 LLM 提取 1-6 字核心概念。

LLM 在这里的角色是"感官"（感知理解），不是"大脑"（记忆/决策/推理）。
"""

from __future__ import annotations

import requests


class ConceptExtractor:
    """概念提取器：从输入文本中提取核心概念。

    用法::

        extractor = ConceptExtractor()
        concept = extractor.extract("这个世界有很多好玩的，你对什么感兴趣？")
        # concept = "好奇"
    """

    DEFAULT_MODEL = "qwen3.5:2b"
    DEFAULT_HOST = "http://localhost:11434"
    SHORT_TEXT_THRESHOLD = 8

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 30.0,
    ) -> None:
        """初始化概念提取器。

        Args:
            model_name: Ollama LLM 模型名
            host: Ollama 服务地址
            timeout: 请求超时秒数
        """
        self._model = model_name
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._llm_available: bool | None = None

    def extract(self, text: str) -> str:
        """从输入文本中提取核心概念。

        短文本（≤8字）直接返回原文。
        长文本用 LLM 提取 1-6 字概念。
        LLM 不可用时回退到截断（取前 8 字）。

        Args:
            text: 输入文本

        Returns:
            核心概念（1-8 字）
        """
        text = text.strip()
        if not text:
            return text

        # 短文本直接返回
        if len(text) <= self.SHORT_TEXT_THRESHOLD:
            return text

        # 长文本用 LLM 提取
        if self._check_llm():
            concept = self._llm_extract(text)
            if concept:
                return concept

        # 回退：取前 8 字
        return text[: self.SHORT_TEXT_THRESHOLD]

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
            self._llm_available = (
                self._model in names
                or f"{self._model}:latest" in names
                or any(n.startswith(self._model) for n in names)
            )
        except Exception:
            self._llm_available = False

        return self._llm_available

    def _llm_extract(self, text: str) -> str:
        """用 LLM 从文本中提取核心概念。

        Args:
            text: 输入文本

        Returns:
            核心概念（1-6 字），或空字符串（失败时）
        """
        prompt = (
            "从以下文本中提取一个核心概念，要求：\n"
            "1. 只输出概念本身，1-6个字\n"
            "2. 不要标点符号，不要解释\n"
            "3. 概念应该是名词或动名词\n\n"
            f"文本：{text}\n"
            "概念："
        )

        try:
            resp = requests.post(
                f"{self._host}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 20,
                    },
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("response", "").strip()
            if not result and data.get("thinking"):
                result = data["thinking"].strip()

            # 清理：去掉标点、换行，取第一行
            result = result.split("\n")[0].strip()
            # 去掉常见标点
            for ch in "，。！？、；：""''（）()【】[]《》<>…—":
                result = result.replace(ch, "")
            result = result.strip()

            # 限制长度
            if len(result) > 10:
                result = result[:6]

            return result if result else ""
        except Exception:
            return ""
