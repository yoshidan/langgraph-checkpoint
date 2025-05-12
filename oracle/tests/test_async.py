# type: ignore
import os
from contextlib import asynccontextmanager
from typing import Any

import pytest
from oracledb import ConnectParams, create_pool_async

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.oracle.aio import AsyncOracleSaver
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
