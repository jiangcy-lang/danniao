# 多模态输入与海马体向量匹配规范

> 本文档为造物主补充规范，与 [`01-hippocampus-cognitive-tree.md`](01-hippocampus-cognitive-tree.md) 共同构成海马体权威定义。  
> **文本 MVP（规则词典 + NetworkX）仍可用于 Step 1 验收**；本规范定义 Step 1.5 → Step 2 的演进目标。

## 1. 海马体底层存储规范

### 1.1 禁止纯字符串节点

- 节点**不得**仅以字符串/concept 作为唯一存储形态。
- **所有节点**（主干 `trunk` 与子节点 `feature`）必须绑定 **高维语义向量（Embedding）**。
- `concept` 等字符串字段仅作**人类可读标签**与 `describe` 输出，不是匹配主键。

### 1.2 物理底座

| 组件 | 职责 |
|------|------|
| **向量库（ChromaDB / FAISS）** | 存储 embedding、相似度检索、持久化 |
| **NetworkX 认知树** | 树状层级、父子边、访问权重（逻辑结构） |
| **双写关联** | 每个 `node_id` 在树与向量库中一一对应 |

**默认选型**：ChromaDB（本地持久化、元数据过滤、工程门槛低）。FAISS 可作为高性能只读检索备选。

### 1.3 节点元数据（向量库 payload）

```json
{
  "node_id": "苹果",
  "concept": "苹果",
  "kind": "trunk",
  "dimension": null,
  "value": null,
  "parent_trunk": null
}
```

子节点示例：`node_id=颜色-红`, `kind=feature`, `dimension=颜色`, `value=红`, `parent_trunk=苹果`。

## 2. 多模态输入的门控处理流程

```text
外部输入（文本 / 图片 / 音频）
        ↓
跨模态对齐编码器（文本嵌入 / CLIP 图像 / 音频编码器）
        ↓
统一语义向量 space
        ↓
海马体向量检索 + 认知树门控（相似度 + 预测误差）
        ↓
激活增强 或 节点繁衍
```

| 模态 | 编码器（推荐） | 说明 |
|------|----------------|------|
| 文本 | Sentence-Transformer / 文本嵌入 API | 与 CLIP 文本塔对齐时优先同空间 |
| 图像 | CLIP ViT | 图像 → 向量，与文本塔共享空间 |
| 音频 | Whisper 嵌入 / 专用音频编码器 | 二期；接口预留 |

**门控前置原则**：非文本输入**不得**直接进入字符串解析器；必须先 `encode → vector`。

## 3. 匹配与生长的联动逻辑

### 3.1 主干匹配（余弦相似度）

对输入向量 `v_in`，在海马体所有 **trunk** 节点中检索：

```text
sim = cosine(v_in, v_trunk)
```

| 条件 | 行为 |
|------|------|
| `sim ≥ τ`（默认 **τ = 0.85**）且**无显著新特征维度** | **仅激活增强**（提升 `access_weight` / 边权重），不生成新节点 |
| `sim ≥ τ` 但存在**显著预测误差（新特征维度）** | 在匹配到的主干下 **节点繁衍**（新 feature 子节点 + embedding 入库） |
| `sim < τ` | 视为**新概念**：创建新孤立主干 + 其 embedding |

### 3.2 新特征维度（预测误差）

向量匹配解决「是不是同一个东西」；**维度繁衍**解决「这个东西有没有新属性」。

| 输入类型 | 新维度判定（工程路径） |
|----------|------------------------|
| 文本 | 结构化解析 `(dimension, value)` + 向量侧验证（子节点 embedding 与输入在维度投影上的差异） |
| 图像 | CLIP 零样本维度探针（如 `"red apple"` vs `"green apple"`）或伴随文本描述 |
| 纯向量 | 与已有子节点 embedding 聚类/残差；残差超过阈值 → 新 feature |

**联动**：相似度高 + 有新维度 → **匹配生长**（挂在已有主干下）；相似度高 + 无新维度 → **只激活**。

### 3.3 与文本 MVP 的关系

| 阶段 | 文本路径 | 向量路径 |
|------|----------|----------|
| Step 1（已完成） | 规则词典 + 字符串门控 | — |
| **Step 1.5（当前）** | 保留验收 | 节点双写 embedding；文本也走向量匹配 |
| Step 2+ | 文本作为标签辅助 | 多模态 CLIP + 相似度门控为主 |

## 4. 验收标准（向量阶段）

1. 创建主干「苹果」时，ChromaDB 中必有对应 embedding，不可仅写 NetworkX。  
2. 文本「苹果」与已有 trunk embedding 余弦相似度 ≥ 0.85 → `routine_activate`，不繁衍。  
3. 高信息输入（新维度）→ 相似度仍匹配同一 trunk，但触发 `spawned_children`。  
4. （可选）图片经 CLIP 编码后与「苹果」trunk 相似度 ≥ 0.85 → 激活，不误建新 trunk。  
5. `describe` 输出不变（标签来自树结构）。

## 5. 依赖与环境

```text
# 核心
chromadb>=0.4
numpy

# 文本嵌入（推荐）
sentence-transformers>=2.2

# 多模态（可选 extras）
torch
open-clip-torch   # 或 transformers + CLIP
```

无 GPU 时可用 CPU + MiniLM；CLIP 图像验收为可选。

## 6. 明确禁止

- 仅用人眼可读字符串做检索匹配  
- 图像/base64 直接当 dict key 写入海马体  
- 相似度低于阈值仍强行合并到最近 trunk（避免概念污染）
