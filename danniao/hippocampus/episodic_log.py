"""第一层存储：原始交互日志（情景记忆）。

SQLite 追加写，永不删除。记录每次交互的完整原文。
类比人脑的情景记忆 —— 像低分辨率录像，细节随时可回放。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodicLog:
    """原始交互日志（SQLite 追加写，永不删除）。

    表结构::

        episodic_log (
            id         TEXT PRIMARY KEY,   -- 交互唯一 ID（幂等）
            timestamp  TEXT,               -- UTC ISO 8601
            text       TEXT,               -- 交互原文
            source     TEXT,               -- 来源（user / search / creator ...）
            metadata   TEXT                -- JSON 序列化的附加元数据
        )
    """

    def __init__(self, path: str = ".danniao_memory/episodic.db") -> None:
        """初始化日志数据库。

        Args:
            path: SQLite 文件路径。``":memory:"`` 使用内存数据库（测试用）。
        """
        self._path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_log (
                id         TEXT PRIMARY KEY,
                timestamp  TEXT NOT NULL,
                text       TEXT NOT NULL,
                source     TEXT NOT NULL DEFAULT 'user',
                metadata   TEXT
            )
        """)
        self._conn.commit()

    def append(
        self,
        interaction_id: str,
        text: str,
        *,
        source: str = "user",
        metadata: dict | None = None,
    ) -> None:
        """追加一条交互记录（幂等：相同 ID 不重复插入）。

        Args:
            interaction_id: 交互唯一标识
            text: 交互原文
            source: 来源标签
            metadata: 附加元数据（JSON 序列化存储）
        """
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        self._conn.execute(
            """INSERT OR IGNORE INTO episodic_log
               (id, timestamp, text, source, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (interaction_id, _utc_now(), text, source, meta_json),
        )
        self._conn.commit()

    def replay(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        """回放交互记录（按时间正序）。

        Args:
            limit: 返回条数上限
            offset: 偏移量

        Returns:
            字典列表，每个字典含 id / timestamp / text / source / metadata
        """
        rows = self._conn.execute(
            """SELECT id, timestamp, text, source, metadata
               FROM episodic_log
               ORDER BY timestamp ASC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            if entry.get("metadata"):
                entry["metadata"] = json.loads(entry["metadata"])
            result.append(entry)
        return result

    def count(self) -> int:
        """返回日志总条数。"""
        row = self._conn.execute("SELECT COUNT(*) FROM episodic_log").fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
