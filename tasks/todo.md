# DanNiao Todo

## 已完成

- [x] 权威文档（`docs/00` / `01` / `02` / `README`）
- [x] Step 1 动态认知树 + 文本门控 + `describe` + REPL
- [x] Step 1.5 向量双写（ChromaDB）+ 余弦匹配门控 + 多模态接口预留
- [x] Git 同步 hook

## 待做

- [ ] Step 2 动力学引擎（激活增强 / 全局衰减）
- [ ] Step 3 好奇心独立模块
- [ ] Step 4 环境交互闭环
- [ ] 可选：安装 `sentence-transformers` / CLIP 做真实多模态验收

## 审查

- 文本验收：`pytest tests/test_cognitive_tree_acceptance.py`
- 向量验收：`pytest tests/test_vector_hippocampus.py`
- 交互演示：`python -m danniao.hippocampus.repl demo`
- 多模态规范：`docs/02-multimodal-embedding-hippocampus.md`
