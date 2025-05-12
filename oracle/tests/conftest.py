# type: ignore
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Sequence

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from oracledb import AsyncConnection, ConnectParams, connect_async

from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    LATEST_VERSION,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
    uuid6,
)
from langgraph.checkpoint.serde.types import TASKS


@pytest.fixture(scope="function")
async def conn() -> AsyncIterator[AsyncConnection]:
    params = ConnectParams(
        host="localhost",
        port=int(os.getenv("PORT", "1521")),
        user="system",
        password="test",
    )
    async with await connect_async(params=params) as conn:
        yield conn


@pytest.fixture(scope="function", autouse=True)
async def clear_test_db(conn: AsyncConnection) -> None:
    """Delete all tables before each test."""
    try:
        await conn.execute("TRUNCATE TABLE checkpoints")
        await conn.execute("TRUNCATE TABLE checkpoint_blobs")
        await conn.execute("TRUNCATE TABLE checkpoint_writes")
        await conn.execute("TRUNCATE TABLE checkpoint_channel_versions")
        await conn.execute("TRUNCATE TABLE checkpoint_migrations")
    except Exception:
        pass


@pytest.fixture
def test_data():
    """Fixture providing test data for checkpoint tests."""

    chkpnt_1: Checkpoint = Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={
            "channel1": [HumanMessage(content="Hello")],
        },
        channel_versions={
            "channel1": "v0.1",
            "channel2": "v0.2",
        },
        versions_seen={
            "__input__": {},
            "__start__": {"__start__": 1},
            "node": {"start:node": 2},
        },
        pending_sends=[],
    )
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            # for backwards compatibility testing
            "thread_ts": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}

    write_1: Sequence = deque([])
    write_2: Sequence = deque(
        [
            ("channel1", [{"content": "Hello channel1", "role": "user"}]),
            (TASKS, [{"content": "Hello channel2", "role": "user"}]),
        ]
    )
    write_3: Sequence = deque([("channel2", [{"content": "Hello3", "role": "user"}])])
    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
        "writes": [write_1, write_2, write_3],
    }


def copy_config_from(config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
    configurable = {
        "thread_id": config["configurable"]["thread_id"],
        "checkpoint_ns": config["configurable"]["checkpoint_ns"],
        "checkpoint_id": checkpoint["id"],
    }
    dst = config.copy()
    dst["configurable"] = configurable
    return dst


def exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}
