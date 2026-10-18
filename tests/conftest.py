"""让测试状态库与开发/生产数据完全隔离。"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path


_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="uejava-agent-tests-"))
os.environ["AGENT_STATE_DB"] = str(_TEST_DATA_DIR / "agent_state.db")
os.environ["BM25_DB_PATH"] = str(_TEST_DATA_DIR / "bm25.db")
atexit.register(shutil.rmtree, _TEST_DATA_DIR, ignore_errors=True)
