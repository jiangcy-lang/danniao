# DanNiao 文档索引（以本目录 Markdown 为准）

> **认知一致性声明**：自 2026-07-22 起，架构与海马体规范以本目录 `.md` 为唯一实现依据。  
> 同目录下历史 `.docx` 仅作背景参考；其中与 Markdown **冲突的条款一律作废**（尤其是 `01` 的扁平预置 8 节点）。

## 权威文档

| 文档 | 内容 |
|------|------|
| [00-architecture.md](00-architecture.md) | 整体架构、模块划界、吸收/输出契约、人脑映射 |
| [01-hippocampus-cognitive-tree.md](01-hippocampus-cognitive-tree.md) | 动态认知树、信息门控、验收标准（取代旧 Step 1 扁平初始化） |

## 历史参考（docx，部分过时）

| 文档 | 状态 |
|------|------|
| 丹鸟创世指南 | 愿景与三大法则仍有效；实现细节以 `00`/`01` 为准 |
| 类脑…设计方案 / _edited | 远期愿景参考 |
| 01–04 模块规范.docx | **部分过时**：以 Markdown 修正为准 |
| Step 1：构建海马体.docx | **过时**：勿再按「预置 8 节点」实现 |
| 1. 认知树的「渐进式生长」机制.docx | 与造物主修正方向一致；细则见 `01-hippocampus-cognitive-tree.md` |

## 实现代码

- 包路径：`danniao/hippocampus/`
- 验收：`tests/test_cognitive_tree_acceptance.py`
