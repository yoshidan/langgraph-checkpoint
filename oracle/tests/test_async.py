# type: ignore
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest
from oracledb import ConnectParams, create_pool_async

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
    get_checkpoint_metadata,
    uuid6,
)
from langgraph.checkpoint.oracle.aio import AsyncOracleSaver
from langgraph.checkpoint.serde.types import TASKS
from langgraph.graph import START, MessagesState, StateGraph
from tests.conftest import copy_config_from, exclude_keys


@asynccontextmanager
async def _base_saver():
    params = ConnectParams(
        host="localhost",
        port=int(os.getenv("PORT", "1521")),
        user="system",
        password="test",
    )
    async with AsyncOracleSaver.from_conn_params(params=params) as checkpointer:
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def _pool_saver(autocommit: bool = False):
    pool = create_pool_async(
        user="system",
        password="test",
        port=int(os.getenv("PORT", "1521")),
        host="localhost",
        min=2,
        max=5,
        increment=1,
    )

    async def acquire():
        conn = await pool.acquire()
        conn.autocommit = autocommit
        return conn

    checkpointer = AsyncOracleSaver(conn=acquire)
    await checkpointer.setup()
    yield checkpointer


@asynccontextmanager
async def _saver(name: str):
    if name == "base":
        async with _base_saver() as saver:
            yield saver
    elif name == "pool":
        async with _pool_saver(autocommit=True) as saver:
            yield saver
    elif name == "tx":
        async with _pool_saver(autocommit=False) as saver:
            yield saver


@pytest.mark.parametrize("saver_name", ["base", "pool", "tx"])
async def test_combined_metadata(saver_name: str, test_data) -> None:
    async with _saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__super_private_key": "super_private_value",
            },
            "metadata": {"run_id": "my_run_id"},
        }
        chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
            "score": None,
        }
        await saver.aput(config, chkpnt, metadata, {})
        checkpoint = await saver.aget_tuple(config)
        assert checkpoint.metadata == {
            **metadata,
            "thread_id": "thread-2",
            "run_id": "my_run_id",
        }


@pytest.mark.parametrize("saver_name", ["base", "pool", "tx"])
async def test_search(saver_name: str, test_data) -> None:
    async with _saver(saver_name) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]
        writes = test_data["writes"]

        await saver.aput(
            configs[0], checkpoints[0], metadata[0], checkpoints[0]["channel_versions"]
        )
        await saver.aput(
            configs[1], checkpoints[1], metadata[1], checkpoints[1]["channel_versions"]
        )
        await saver.aput_writes(
            copy_config_from(configs[1], checkpoints[1]), writes[1], "task2", "path2"
        )
        await saver.aput(
            configs[2], checkpoints[2], metadata[2], checkpoints[2]["channel_versions"]
        )
        await saver.aput_writes(
            copy_config_from(configs[2], checkpoints[2]), writes[2], "task3", "path3"
        )

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == {
            **exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

        search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
        assert len(search_results_3) == 3

        search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
        ]
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


@pytest.mark.parametrize("saver_name", ["base", "pool", "tx"])
async def test_get_tuple(saver_name: str, test_data) -> None:
    async with _saver(saver_name) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]
        writes = test_data["writes"]

        await saver.aput(
            configs[0], checkpoints[0], metadata[0], checkpoints[0]["channel_versions"]
        )
        await saver.aput(
            configs[1], checkpoints[1], metadata[1], checkpoints[1]["channel_versions"]
        )
        await saver.aput_writes(
            copy_config_from(configs[1], checkpoints[1]), writes[1], "task2", "path2"
        )
        await saver.aput(
            configs[2], checkpoints[2], metadata[2], checkpoints[2]["channel_versions"]
        )
        await saver.aput_writes(
            copy_config_from(configs[2], checkpoints[2]), writes[2], "task3", "path3"
        )

        result0 = await saver.aget_tuple(configs[0])
        assert result0 is None

        result1 = await saver.aget_tuple(copy_config_from(configs[0], checkpoints[0]))
        assert len(result1.pending_writes) == 0
        assert result1.config == copy_config_from(configs[0], checkpoints[0])
        assert result1.checkpoint == checkpoints[0]
        # having thread_ts in result1.metadata
        assert result1.metadata == get_checkpoint_metadata(configs[0], metadata[0])
        # thread_ts is not in the parent_config
        parent_config = configs[0].copy()
        parent_config["configurable"] = {
            "thread_id": configs[0]["configurable"]["thread_id"],
            "checkpoint_ns": configs[0]["configurable"]["checkpoint_ns"],
            "checkpoint_id": configs[0]["configurable"]["thread_ts"],
        }
        assert result1.parent_config == parent_config

        result2 = await saver.aget_tuple(copy_config_from(configs[1], checkpoints[1]))
        assert len(result2.pending_writes) == 2
        assert result2.parent_config == configs[1]
        assert result2.config == copy_config_from(configs[1], checkpoints[1])
        assert result2.checkpoint == checkpoints[1]
        assert result2.metadata == get_checkpoint_metadata(result2.config, metadata[1])

        result3 = await saver.aget_tuple(copy_config_from(configs[2], checkpoints[2]))
        assert len(result3.pending_writes) == 1
        assert result3.parent_config == configs[2]
        assert result3.config == copy_config_from(configs[2], checkpoints[2])
        assert result3.checkpoint == checkpoints[2]
        assert result3.metadata == get_checkpoint_metadata(result3.config, metadata[2])


@pytest.mark.parametrize("saver_name", ["base", "pool", "tx"])
async def test_null_chars(saver_name: str, test_data) -> None:
    async with _saver(saver_name) as saver:
        config = await saver.aput(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {"my_key": "\x00abc"},
            {},
        )
        assert (await saver.aget_tuple(config)).metadata["my_key"] == "abc"  # type: ignore
        assert [c async for c in saver.alist(None, filter={"my_key": "abc"})][
            0
        ].metadata["my_key"] == "abc"


@pytest.mark.parametrize("saver_name", ["base", "pool", "tx"])
async def test_large_data_over_4000_bytes(saver_name: str) -> None:
    """Test that checkpoints with data exceeding 4000 bytes can be stored and retrieved.

    This test verifies the fix for ORA-40478 error by ensuring that JSON_ARRAYAGG
    returns CLOB instead of VARCHAR2, which has a 4000-byte limit.
    """
    async with _saver(saver_name) as saver:
        # Create a large message that exceeds 4000 bytes
        # Each channel will have multiple large messages
        large_content = "x" * 2000  # 2KB per message

        # Create checkpoint with large channel values (will be aggregated by JSON_ARRAYAGG)
        config_1: Any = {
            "configurable": {
                "thread_id": "thread-large-1",
                "checkpoint_ns": "",
            }
        }

        chkpnt_1: Checkpoint = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-2)),
            ts=datetime.now(timezone.utc).isoformat(),
            channel_values={
                "channel1": [{"content": large_content, "role": "user"}],
                "channel2": [{"content": large_content, "role": "assistant"}],
                "channel3": [{"content": large_content, "role": "system"}],
            },
            channel_versions={
                "channel1": "v1",
                "channel2": "v1",
                "channel3": "v1",
            },
            versions_seen={
                "__input__": {},
                "__start__": {"__start__": 1},
            },
            pending_sends=[],
        )

        metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 1,
            "writes": {"large_key": large_content},
            "description": large_content,
        }

        # Store first checkpoint
        config_result_1 = await saver.aput(
            config_1, chkpnt_1, metadata_1, chkpnt_1["channel_versions"]
        )

        # Verify first checkpoint can be retrieved
        result_1 = await saver.aget_tuple(config_result_1)
        assert result_1 is not None
        assert (
            result_1.checkpoint["channel_values"]["channel1"][0]["content"]
            == large_content
        )
        assert (
            result_1.checkpoint["channel_values"]["channel2"][0]["content"]
            == large_content
        )
        assert (
            result_1.checkpoint["channel_values"]["channel3"][0]["content"]
            == large_content
        )
        assert result_1.metadata["description"] == large_content

        # Create second checkpoint with even more data to ensure cumulative data > 4000 bytes
        config_2: Any = {
            "configurable": {
                "thread_id": "thread-large-1",
                "checkpoint_id": chkpnt_1["id"],
                "checkpoint_ns": "",
            }
        }

        chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 2)
        chkpnt_2["channel_values"] = {
            "channel1": [{"content": large_content + "1", "role": "user"}],
            "channel2": [{"content": large_content + "2", "role": "assistant"}],
            "channel3": [{"content": large_content + "3", "role": "system"}],
            "channel4": [{"content": large_content + "4", "role": "user"}],
        }
        chkpnt_2["channel_versions"] = {
            "channel1": "v2",
            "channel2": "v2",
            "channel3": "v2",
            "channel4": "v2",
        }

        metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 2,
            "writes": {"large_key": large_content + "second"},
            "description": large_content + "second",
        }

        # Add large pending writes (tests PENDING_WRITES aggregation)
        large_writes: Any = [
            ("channel1", [{"content": large_content + "write1", "role": "user"}]),
            ("channel2", [{"content": large_content + "write2", "role": "assistant"}]),
            (TASKS, [{"content": large_content + "task", "role": "system"}]),
        ]

        # Store second checkpoint
        config_result_2 = await saver.aput(
            config_2, chkpnt_2, metadata_2, chkpnt_2["channel_versions"]
        )

        # Add pending writes
        await saver.aput_writes(
            config_result_2, large_writes, "task-large", "path-large"
        )

        # Verify second checkpoint can be retrieved (this would fail with ORA-40478 before the fix)
        result_2 = await saver.aget_tuple(config_result_2)
        assert result_2 is not None
        assert (
            result_2.checkpoint["channel_values"]["channel1"][0]["content"]
            == large_content + "1"
        )
        assert (
            result_2.checkpoint["channel_values"]["channel2"][0]["content"]
            == large_content + "2"
        )
        assert (
            result_2.checkpoint["channel_values"]["channel3"][0]["content"]
            == large_content + "3"
        )
        assert (
            result_2.checkpoint["channel_values"]["channel4"][0]["content"]
            == large_content + "4"
        )
        assert result_2.metadata["description"] == large_content + "second"

        # Verify pending writes were stored correctly
        assert len(result_2.pending_writes) == 3
        pending_contents = [w[2][0]["content"] for w in result_2.pending_writes]
        assert large_content + "write1" in pending_contents
        assert large_content + "write2" in pending_contents
        assert large_content + "task" in pending_contents

        # List all checkpoints for this thread (tests aggregation across multiple checkpoints)
        all_checkpoints = [
            c
            async for c in saver.alist(
                {"configurable": {"thread_id": "thread-large-1"}}
            )
        ]
        assert len(all_checkpoints) == 2

        # Verify both checkpoints have correct large data
        for checkpoint_tuple in all_checkpoints:
            assert checkpoint_tuple.checkpoint is not None
            assert checkpoint_tuple.metadata is not None
            # Each checkpoint should have large channel values
            for channel_values in checkpoint_tuple.checkpoint[
                "channel_values"
            ].values():
                assert len(channel_values[0]["content"]) >= 2000


@pytest.mark.parametrize("saver_name", ["base", "pool", "tx"])
async def test_multiple_invocations_same_thread(saver_name: str) -> None:
    """Test that graph nodes execute correctly across multiple invocations with same thread_id.

    This test verifies the fix for an issue where nodes would stop executing after
    a few iterations when using the same thread_id repeatedly. The root cause was
    that get_next_version returned non-unique version numbers, causing LangGraph
    to incorrectly detect nodes as "already executed".
    """
    execution_log: list[str] = []

    def node1(state: MessagesState):
        execution_log.append("node1")
        return {"messages": f"msg1-{random.randint(0, 100)}"}

    def node2(state: MessagesState):
        execution_log.append("node2")
        return {"messages": f"msg2-{random.randint(0, 100)}"}

    builder = StateGraph(MessagesState)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    thread_id = f"test-multi-invoke-{datetime.now().isoformat()}"
    config = {"configurable": {"thread_id": thread_id}}

    async with _saver(saver_name) as saver:
        graph = builder.compile(checkpointer=saver)

        # Run multiple iterations with the same thread_id
        for i in range(6):
            execution_log.clear()

            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "test message"}]},
                config,
            )

            # Both nodes should execute on every iteration
            assert "node1" in execution_log, f"node1 did not execute on iteration {i}"
            assert "node2" in execution_log, f"node2 did not execute on iteration {i}"
            assert result is not None, f"Result was None on iteration {i}"

            # Messages should accumulate: initial + (node1 + node2) per iteration
            expected_count = (
                1 + (i + 1) * 2 + i
            )  # input + nodes output + previous inputs
            assert (
                len(result["messages"]) == expected_count
            ), f"Expected {expected_count} messages on iteration {i}, got {len(result['messages'])}"
