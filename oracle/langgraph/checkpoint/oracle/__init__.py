import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from oracledb import ConnectParams, Cursor, connect

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.oracle._internal import Conn, get_connection
from langgraph.checkpoint.oracle.base import BaseOracleSaver
from langgraph.checkpoint.oracle.utils import (
    decode_channel_values,
    decode_channel_versions,
    decode_pending_sends,
    decode_pending_writes,
)
from langgraph.checkpoint.serde.base import SerializerProtocol


class OracleSaver(BaseOracleSaver):
    """Checkpointer that stores checkpoints in an Oracle database."""

    lock: threading.Lock

    def __init__(
        self,
        conn: Conn,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = threading.Lock()

    @classmethod
    @contextmanager
    def from_conn_params(cls, params: ConnectParams) -> Iterator["OracleSaver"]:
        """Create a new OracleSaver instance from a connection string.

        Args:
            params: The Oracle connection params:

        Returns:
            OracleSaver: A new OracleSaver instance.
        """
        conn = connect(params=params)
        conn.autocommit = True
        try:
            yield cls(conn)
        finally:
            conn.close()

    def setup(self) -> None:
        """Set up the checkpoint database.

        This method creates the necessary tables in the Oracle database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        with self._cursor() as cur:
            # Create migrations table if it doesn't exist
            _execute_ddl(self.MIGRATIONS[0], cur)

            # Get current version
            cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC FETCH FIRST 1 ROW ONLY"
            )
            row = cur.fetchone()
            version = 0 if row is None else row[0]

            # Run migrations in a transaction
            for v, migration in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                _execute_ddl(migration, cur)
                cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Oracle database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.

        Examples:
            >>> from langgraph.checkpoint.oracle import OracleSaver
            >>> params = {...}  # Oracle connection parameters
            >>> with OracleSaver.from_conn_params(params) as memory:
            ...     config = {"configurable": {"thread_id": "1"}}
            ...     checkpoints = list(memory.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]

            >>> config = {"configurable": {"thread_id": "1"}}
            >>> before = {"configurable": {"checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"}}
            >>> with OracleSaver.from_conn_params(params) as memory:
            ...     checkpoints = list(memory.list(config, before=before))
            >>> print(checkpoints)
            [CheckpointTuple(...), ...]
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY c.checkpoint_id DESC"
        if limit:
            query += f" FETCH FIRST {limit} ROWS ONLY"
        for _checkpoint in self._fetch_checkpoint(query, args):
            checkpoint = _checkpoint
            yield checkpoint

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Oracle database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Examples:

            Basic:
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)

            With timestamp:

            >>> config = {
            ...    "configurable": {
            ...        "thread_id": "1",
            ...        "checkpoint_ns": "",
            ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            ...    }
            ... }
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)
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

        for checkpoint in self._fetch_checkpoint(self.SELECT_SQL + where, args):
            return checkpoint

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint in the database.

        Args:
            config: The config to use for storing the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: The metadata to store with the checkpoint.
            new_versions: The new versions of the channels.

        Returns:
            RunnableConfig: The config with the checkpoint ID.

        Examples:

            >>> from langgraph.checkpoint.oracle import OracleSaver
            >>> params = {...}  # Oracle connection parameters
            >>> with OracleSaver.from_conn_params(params) as memory:
            ...     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            ...     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "channel_values": {"key": "value"}}
            ...     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
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

        with self._cursor() as cur:
            # Store checkpoint blobs
            blobs = self._dump_blobs(
                thread_id,
                checkpoint_ns,
                copy.pop("channel_values"),  # type: ignore[misc]
                new_versions,
            )
            if len(blobs) > 0:
                cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, blobs)

            # Store checkpoint channel version
            channel_versions = [
                (thread_id, checkpoint_ns, checkpoint["id"], k, v)
                for k, v in copy.pop("channel_versions").items()  # type: ignore[misc]
            ]
            if len(channel_versions) > 0:
                cur.executemany(
                    self.UPSERT_CHECKPOINT_CHANNEL_VERSIONS_SQL, channel_versions
                )

            # Store checkpoint
            cur.execute(
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

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes in the database.

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
        params = self._dump_writes(
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        if len(params) > 0:
            with self._cursor() as cur:
                cur.executemany(query, params)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = :1",
                [thread_id],
            )
            cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = :1",
                [thread_id],
            )
            cur.execute(
                "DELETE FROM checkpoint_channel_versions WHERE thread_id = :1",
                [thread_id],
            )
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = :1",
                [thread_id],
            )

    @contextmanager
    def _cursor(
        self,
    ) -> Iterator[Cursor]:
        """Get a cursor for executing SQL statements.

        Returns:
            Iterator[Cursor]: A cursor for executing SQL statements.
        """
        with get_connection(self.conn) as conn:
            if conn.autocommit is False:
                with self.lock:
                    try:
                        with conn.cursor() as cur:
                            yield cur
                        conn.commit()
                    except:
                        conn.rollback()
                        raise
            else:
                with self.lock, conn.cursor() as cur:
                    yield cur

    def _fetch_checkpoint(self, query: str, args: Any) -> Iterator[CheckpointTuple]:
        """
            Execute a query to fetch checkpoint data from the database and yield it as CheckpointTuple objects.

        Args:
            query (str): The SQL query to execute.
            args (Any): The arguments to pass to the SQL query.

        Yields:
            Iterator[CheckpointTuple]: An iterator of CheckpointTuple objects, each representing a checkpoint
            retrieved from the database.
        """
        with self._cursor() as cur:
            cur.execute(query, args)
            columns = [col[0] for col in cur.description]
            for value in cur:
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
                    self._load_checkpoint(
                        self.jsonplus_serde.loads(row_dict["CHECKPOINT"].read()),
                        decode_channel_values(
                            row_dict["CHANNEL_VALUES"].read()
                            if row_dict["CHANNEL_VALUES"]
                            else None,
                            self.jsonplus_serde,
                        ),
                        decode_pending_sends(
                            row_dict["PENDING_SENDS"].read()
                            if row_dict["PENDING_SENDS"]
                            else None,
                            self.jsonplus_serde,
                        ),
                        decode_channel_versions(
                            row_dict["CHANNEL_VERSIONS"].read()
                            if row_dict["CHANNEL_VERSIONS"]
                            else None,
                            self.jsonplus_serde,
                        ),
                    ),
                    self._load_metadata(
                        self.jsonplus_serde.loads(row_dict["METADATA"].read())
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
                    self._load_writes(
                        decode_pending_writes(
                            row_dict["PENDING_WRITES"].read()
                            if row_dict["PENDING_WRITES"]
                            else None,
                            self.jsonplus_serde,
                        )
                    ),
                )


def _execute_ddl(ddl: str, cur: Cursor) -> None:
    """Execute a DDL statement.

    Args:
        ddl: The DDL statement to execute.
        curl: Cursor
    """
    try:
        cur.execute(ddl)
    except Exception as e:
        if "ORA-00955" not in str(e):  # Object already exists
            raise
