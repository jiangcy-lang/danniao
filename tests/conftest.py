"""pytest 全局配置。

在测试会话开始前清理所有残留的 ChromaDB 测试目录。
原因：_sync_from_store 修复后，ChromaDB 持久化数据会被正确加载到 NetworkX，
如果上一轮测试的残留数据没清理干净（Windows 文件锁），会导致测试断言失败。
"""

import glob
import shutil


def pytest_sessionstart(session):
    """测试会话开始前清理所有 .test_chroma_* 目录。"""
    for pattern in (".test_chroma_*", ".test_*_chroma", ".test_*_memory.db"):
        for path in glob.glob(pattern):
            shutil.rmtree(path, ignore_errors=True)
