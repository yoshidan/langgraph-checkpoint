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
        self.ttl_config = ttl
        self._ttl_sweeper_task: Optional[asyncio.Task[None]] = None
        self._ttl_stop_event = asyncio.Event()

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

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = self._group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with self._cursor() as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )
            if SearchOp in grouped_ops:
                await self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]), results, cur
                )
            if ListNamespacesOp in grouped_ops:
                await self._batch_list_namespaces_ops(
                    cast(Sequence[tuple[int, ListNamespacesOp]], grouped_ops[ListNamespacesOp]), results, cur
                )
            if PutOp in grouped_ops:
                await self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )
        return results

    async def setup(self) -> None:
        async with self._cursor() as cur:
            # テーブル作成（PLSQLブロックを使わずDDLを直接実行）
            try:
                await cur.execute("""
                    CREATE TABLE store (
                        prefix VARCHAR2(255) NOT NULL,
                        key VARCHAR2(255) NOT NULL,
                        value CLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NULL,
                        ttl_minutes NUMBER NULL,
                        PRIMARY KEY (prefix, key)
                    )
                """)
            except oracledb.DatabaseError as e:
                if "ORA-00955" not in str(e):
                    raise
            try:
                await cur.execute("CREATE INDEX store_prefix_idx ON store (prefix)")
            except oracledb.DatabaseError as e:
                if "ORA-00955" not in str(e):
                    raise
            try:
                await cur.execute("CREATE INDEX idx_store_expires_at ON store (expires_at)")
            except oracledb.DatabaseError as e:
                if "ORA-00955" not in str(e):
                    raise

            # ベクトルテーブル
            if self.index_config:
                try:
                    await cur.execute(f"""
                        CREATE TABLE store_vectors (
                            prefix VARCHAR2(255) NOT NULL,
                            key VARCHAR2(255) NOT NULL,
                            field_name VARCHAR2(255) NOT NULL,
                            embedding VECTOR,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (prefix, key, field_name)
                        )
                    """)
                except oracledb.DatabaseError as e:
                    if "ORA-00955" not in str(e):
                        raise
                try:
                    await cur.execute("CREATE INDEX store_vectors_embedding_idx ON store_vectors (embedding) INDEXTYPE IS VECTOR_INDEX")
                except oracledb.DatabaseError as e:
                    if "ORA-00955" not in str(e):
                        raise

