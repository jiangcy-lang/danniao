# DanNiao Todo

## 本轮：海马体动态认知树

- [x] 权威文档对齐（`docs/00` / `01` / `README`）
- [x] 修订实现计划（含最小 `describe` 输出）
- [x] 实现 `danniao/hippocampus`（树 + 特征 + 门控）
- [x] 验收测试通过
- [x] 更新 lessons / 本文件审查结论

## 审查

- 文档：Markdown 为权威；旧 docx 扁平预置作废；划界与输出契约已写清。
- 代码：空图起步；门控区分常规激活 / 新维度繁衍；`describe` 提供最小表达。
- 验收：`pytest tests/test_cognitive_tree_acceptance.py` 通过（苹果孤立 → 繁衍 → 不繁衍 → describe）。
- 未做：Step 2–4、docx 二进制改写、LLM 抽取。
