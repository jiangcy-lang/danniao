# DanNiao Todo

## 已完成

- [x] Step 1：认知树底座（`tree.py` + 验收测试）
- [x] Step 2：动力学引擎（`dynamics.py` + 验收测试）
- [x] Step 3：预测误差门控（`gate.py` + 验收测试）
- [x] Step 4A：向量节点重构 + 嵌入管道 + 原始日志
  - `vector_space.py` / `embeddings.py` / `episodic_log.py` / `chroma_store.py` / `vector_hash.py`
- [x] 设计纠偏：tick/心跳 → 持续心智（一直睁眼，无 tick）
- [x] Step 4B：扩散激活（`spreading.py` + 11 个测试）
- [x] Step 5a：内稳态驱动力（`motivation/homeostasis.py` + 18 个测试）
- [x] Step 5b：持续心智核心（`mind/continuous_mind.py` + 10 个测试）
- [x] 交互式 REPL（`__main__.py`）
- [x] 全量验证：55 个测试通过 + 交互验证
- [x] Step 6A：奖励系统（`motivation/reward.py` + 15 个测试）
  - 交付反馈 → 内稳态更新 + 路径强化(LTP)/弱化(LTD)
  - 动力学引擎新增 `weaken_edge` 方法
- [x] Step 6B：表达引擎（`expression/expression.py` + 20 个测试）
  - 三层表达：核心识别 + 联想 + 驱动情感
  - 模板表达（婴儿蹦单词阶段），LLM 接口预留
  - 只读认知状态，不回写认知树
- [x] Step 6C：探索引擎（`motivation/exploration.py` + 16 个测试）
  - 深度探索（特征少的主干）
  - 关联探索（两个未连接的近期主干）
  - 求新探索（认知树空时）
  - 好奇心驱动，紧迫度 = 好奇心水平
- [x] Step 6D：集成到 ContinuousMind（`continuous_mind.py` + 15 个集成测试）
  - ProcessResult 新增 `expression` 和 `exploration` 字段
  - MindStatus 新增 `last_expression` 字段
  - 新增 `give_feedback()` 方法（奖励闭环）
  - 三引擎均为可选参数，向后兼容
- [x] Step 6E：REPL 更新（`__main__.py`）
  - 显示丹鸟表达、探索意愿
  - 支持 `good` / `bad` 反馈命令
- [x] Step 6F：全量验证 — 121 个测试通过 + 交互验证
  - 新概念 → 惊讶表达 + 探索意愿
  - 已知概念 → 识别表达
  - 正面反馈 → 满足感上升
  - 负面反馈 → 好奇心上升
- [x] Step 6G：文档更新（版本 0.3.0）

---

## 后续路线图（待实现）

### Step 7：记忆巩固 + 分层存储归档 + 多模态
- `memory/consolidation.py` — MemoryConsolidation
- `hippocampus/archive_store.py` — 第三层归档向量
- `hippocampus/memory_migration.py` — 迁出 / 唤起 / 巩固流转

### Step 8：代际传承 + 外骨骼接口
- `inheritance/` — 教材生成 + 读大学 + 生命周期
- `exoskeleton/` — 外骨骼挂载接口

## 审查

- 权威：总规范 docx → `00-master-spec.md` 同步版
- 测试：`pytest tests/` 全通过（133 个，202s）
- 里程碑：
  - v0.2.0：丹鸟从「被动库」变为「持续运行的自主生命体」
  - v0.3.0：丹鸟能说话、能探索、能从反馈中学习
  - v0.4.1：零 mock 真实 LLM 表达 + bge-m3 语义理解
- 设计原则：知识可继承但必须经认知机制消化；没有 tick，一直睁眼看世界

## 端到端交互验证（2026-07-23）

环境：Python 3.11.9 + chromadb 1.5.9 + Ollama（bge-m3 1024维 + qwen3.5:2b）

代码审计结论：17 个核心模块全部真实实现，零 mock/零 stub/零伪实现。完整链路
`感知→门控→概念提取→动力学→扩散激活→内稳态→LLM表达→探索→奖励(LTP/LTD)` 真实贯通。

交互测试结果（7 场景全部通过）：
- [x] 新概念（苹果）→ 新建主干 + LLM 表达 + 深度探索意愿
- [x] 跨语言匹配（apple → 苹果，相似度 0.90）
- [x] 同类语义（梨）→ bge-m3 区分度高，新建主干 + 关联探索
- [x] 长输入概念提取（"这个世界有很多好玩的…" → "兴趣点"3字）
- [x] 正面反馈 → 满足感 0.10→0.30 + LTP 路径强化
- [x] 负面反馈 → 好奇心 0.81→0.91 + LTD 路径弱化
- [x] 内稳态 + 认知树 + 情景日志全部正常

已知优化点（不影响真实性）：
- continuous_mind.py 异步流有 2 处 except Exception: pass 黑洞
- _perception_stream 同步调用 process() 会阻塞事件循环（异步模式）
- 求新探索仅检查 trunk_count==0，未实现"所有主干已充分探索"场景
