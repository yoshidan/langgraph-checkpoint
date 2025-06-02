import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, Union, cast

import orjson
import oracledb

from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.oracle.base import (
    BaseOracleStore,
    OracleIndexConfig,
    TTLConfig,
    _ensure_index_config,
)

logger = logging.getLogger(__name__)

class AsyncOracleStore(AsyncBatchedBaseStore, BaseOracleStore):
    """Async Oracle 23ai store (with vector search support)"""

    def __init__(
        self,
        conn: oracledb.Connection,
        *,
        deserializer: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]] = None,
        index: Optional[OracleIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        super().__init__(conn=conn, deserializer=deserializer, index=index, ttl=ttl)
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: Optional[OracleIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> AsyncIterator["AsyncOracleStore"]:
        conn = await oracledb.connect(conn_string)
        try:
            yield cls(conn=conn, index=index, ttl=ttl)
        finally:
            await conn.close()

    # Implement methods similar to Postgres aio.py for Oracle
    # For example: abatch, setup, sweep_ttl, start_ttl_sweeper, stop_ttl_sweeper, etc.

