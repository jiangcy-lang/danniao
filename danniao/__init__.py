"""DanNiao — 类脑数字生命体。

架构概览：
- hippocampus：认知树 + 门控 + 动力学 + 扩散激活（记忆底座）
  - OllamaEmbedding：bge-m3 本地嵌入（1024维，多语言）
  - ChromaVectorStore：ChromaDB 向量存储（维度自适应）
- motivation：内稳态驱动力 + 奖励系统 + 探索引擎（动机层）
- expression：模板表达 + LLM 表达（qwen3.5:2b，丹鸟的「嘴巴」）
- mind：持续心智核心（一直睁眼，无 tick）

v0.4.1 修复：
- 嵌入模型 nomic-embed-text → bge-m3（修复中文短文本语义坍缩）
- 门控阈值 0.5 → 0.65（适配 bge-m3 相似度分布）
- ChromaDB 维度自适应：初始化时检查，避免 Windows 文件锁
"""

__version__ = "0.4.1"
