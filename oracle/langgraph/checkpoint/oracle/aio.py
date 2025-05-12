import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from oracledb import AsyncCursor, ConnectParams, connect_async

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.oracle import (
    decode_channel_values,
    decode_channel_versions,
    decode_pending_sends,
    decode_pending_writes,
)
from langgraph.checkpoint.oracle._ainternal import Conn, get_connection
from langgraph.checkpoint.oracle.base import BaseOracleSaver
from langgraph.checkpoint.serde.base import SerializerProtocol


class AsyncOracleSaver(BaseOracleSaver):
    """Asynchronous checkpointer that stores checkpoints in an Oracle database."""

    def __init__(
        self,
        conn: Conn,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_params(
        cls, params: ConnectParams
    ) -> AsyncIterator["AsyncOracleSaver"]:
        """Create a new AsyncOracleSaver instance from a connection string.

        Args:
            params: The Oracle connection params:

        Returns:
            AsyncOracleSaver: A new AsyncOracleSaver instance.
        """
        conn = await connect_async(params=params)
        conn.autocommit = True
        try:
            yield cls(conn)
        finally:
            await conn.close()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the Oracle database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        async with self._cursor() as cur:
            # Create migrations table if it doesn't exist
            await _execute_ddl(self.MIGRATIONS[0], cur)

            # Get current version
            await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC FETCH FIRST 1 ROW ONLY"
            )
            row = await cur.fetchone()
            version = 0 if row is None else row[0]

            # Run migrations in a transaction
            for v, migration in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                await _execute_ddl(migration, cur)
                await cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Oracle database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            AsyncIterator[CheckpointTuple]: An async iterator of checkpoint tuples.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY c.checkpoint_id DESC"
        if limit:
            query += f" FETCH FIRST {limit} ROWS ONLY"

        async for checkpoint in self._fetch_checkpoint(query, args):
            yield checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Oracle database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            args = [thread_id, checkpoint_ns, checkpoint_id]
            where = "WHERE c.thread_id = :1 AND c.checkpoint_ns = NVL(:2, '__|default|__') AND c.checkpoint_id = :3"
        else:
            args = [thread_id, checkpoint_ns]
            where = "WHERE c.thread_id = :1 AND c.checkpoint_ns = NVL(:2, '__|default|__') ORDER BY c.checkpoint_id DESC FETCH FIRST 1 ROW ONLY"

        async for checkpoint in self._fetch_checkpoint(self.SELECT_SQL + where, args):
            return checkpoint

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint in the database asynchronously.

        Args:
            config: The config to use for storing the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: The metadata to store with the checkpoint.
            new_versions: The new versions of the channels.

        Returns:
            RunnableConfig: The config with the checkpoint ID.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self._cursor() as cur:
            # Store checkpoint blobs
            blobs = await asyncio.to_thread(
                self._dump_blobs,
                thread_id,
                checkpoint_ns,
                copy.pop("channel_values"),  # type: ignore[misc]
                new_versions,
            )
            if len(blobs) > 0:
                await cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, blobs)

            # Store checkpoint channel version
            channel_versions = [
                (thread_id, checkpoint_ns, checkpoint["id"], k, v)
                for k, v in copy.pop("channel_versions").items()  # type: ignore[misc]
            ]
            if len(channel_versions) > 0:
                await cur.executemany(
                    self.UPSERT_CHECKPOINT_CHANNEL_VERSIONS_SQL, channel_versions
                )

            # Store checkpoint
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    self.jsonplus_serde.dumps(self._dump_checkpoint(copy))
                    .decode()
                    .replace("\\u0000", ""),
                    self._dump_metadata(get_checkpoint_metadata(config, metadata)),
                ),
            )

        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes in the database asynchronously.

        Args:
            config: The config to use for storing the writes.
            writes: The writes to store.
            task_id: The ID of the task.
            task_path: The path of the task.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        if len(params) > 0:
            async with self._cursor() as cur:
                await cur.executemany(query, params)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread asynchronously.

        Args:
            thread_id: The ID of the thread to delete.
        """
        async with self._cursor() as cur:
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = :1",
                [thread_id],
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = :1",
                [thread_id],
            )
            await cur.execute(
                "DELETE FROM checkpoint_channel_versions WHERE thread_id = :1",
                [thread_id],
            )
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = :1",
                [thread_id],
            )

    @asynccontextmanager
    async def _cursor(
        self,
    ) -> AsyncIterator[AsyncCursor]:
        """Get a cursor for executing SQL statements.

        Returns:
            Iterator[Cursor[DictRow]]: A cursor for executing SQL statements.
        """
        async with get_connection(self.conn) as conn:
            if conn.autocommit is False:
                async with self.lock:
                    try:
                        with conn.cursor() as cur:
                            yield cur
                        await conn.commit()
                    except:
                        await conn.rollback()
                        raise
            else:
                async with self.lock:
                    with conn.cursor() as cur:
                        yield cur

    async def _fetch_checkpoint(
        self, query: str, args: list[Any]
    ) -> AsyncIterator[CheckpointTuple]:
        """
            Execute a query to fetch checkpoint data from the database and yield it as CheckpointTuple objects.

        Args:
            query (str): The SQL query to execute.
            args (Any): The arguments to pass to the SQL query.

        Yields:
            Iterator[CheckpointTuple]: An iterator of CheckpointTuple objects, each representing a checkpoint
            retrieved from the database.
        """
        async with self._cursor() as cur:
            await cur.execute(query, args)
            columns = [col[0] for col in cur.description]
            async for value in cur:
                row_dict = dict(zip(columns, value))
                checkpoint_ns = row_dict["CHECKPOINT_NS"]
                if checkpoint_ns is None:
                    checkpoint_ns = ""
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": row_dict["THREAD_ID"],
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row_dict["CHECKPOINT_ID"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        self.jsonplus_serde.loads(await row_dict["CHECKPOINT"].read()),
                        decode_channel_values(
                            row_dict["CHANNEL_VALUES"], self.jsonplus_serde
                        ),
                        decode_pending_sends(
                            row_dict["PENDING_SENDS"], self.jsonplus_serde
                        ),
                        decode_channel_versions(
                            row_dict["CHANNEL_VERSIONS"], self.jsonplus_serde
                        ),
                    ),
                    self._load_metadata(
                        self.jsonplus_serde.loads(await row_dict["METADATA"].read())
                    ),
                    (
                        {
                            "configurable": {
                                "thread_id": row_dict["THREAD_ID"],
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": row_dict["PARENT_CHECKPOINT_ID"],
                            }
                        }
                        if row_dict["PARENT_CHECKPOINT_ID"]
                        else None
                    ),
                    await asyncio.to_thread(
                        self._load_writes,
                        decode_pending_writes(
                            row_dict["PENDING_WRITES"], self.jsonplus_serde
                        ),
                    ),
                )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit: Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # noqa: F821
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Postgres database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Postgres database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


async def _execute_ddl(ddl: str, cur: AsyncCursor) -> None:
    """Execute a DDL statement.

    Args:
        ddl: The DDL statement to execute.
        cur: Cursor
    """
    try:
        await cur.execute(ddl)
    except Exception as e:
        if "ORA-00955" not in str(e):  # Object already exists
            raise


__all__ = ["AsyncOracleSaver"]
