"""向量确定性哈希：从 numpy 向量生成稳定的 node_id。

同一向量（相同值）在任何进程、任何平台都产出相同 ID。
用于「向量即节点」架构中节点的唯一身份标识。
"""

from __future__ import annotations

import hashlib

import numpy as np


def hash_vector(
    vec: np.ndarray,
    *,
    prefix: str = "v",
    length: int = 16,
) -> str:
    """从 numpy 向量确定性生成稳定 node_id。

    统一 dtype 为 float32 并确保连续内存布局，
    然后对字节做 SHA-256 哈希，截断到指定长度。

    Args:
        vec: 输入向量（任意 dtype 和形状均可，会先 flatten）
        prefix: ID 前缀，默认 ``v_`` 区分于旧字符串 ID
        length: 十六进制截断长度，默认 16（碰撞概率极低）

    Returns:
        形如 ``v_a3f1b9c2d4e5f6a7`` 的稳定 ID
    """
    v = np.ascontiguousarray(vec, dtype=np.float32).ravel()
    h = hashlib.sha256(v.tobytes()).hexdigest()
    return f"{prefix}_{h[:length]}"
