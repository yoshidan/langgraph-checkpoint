from collections.abc import Sequence
from typing import Any, Generator, Optional, cast

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS, ChannelProtocol

MetadataInput = Optional[dict[str, Any]]

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE checkpoint_migrations (
    v NUMBER PRIMARY KEY
)""",
    """CREATE TABLE checkpoints (
    thread_id VARCHAR2(128) NOT NULL,
    checkpoint_ns VARCHAR2(64) NOT NULL,
    checkpoint_id VARCHAR2(64) NOT NULL,
    parent_checkpoint_id VARCHAR2(64),
    type VARCHAR2(64),
    checkpoint CLOB NOT NULL,
    metadata CLOB NOT NULL,
    CONSTRAINT pk_checkpoints PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)""",
    """CREATE TABLE checkpoint_blobs (
    thread_id VARCHAR2(128) NOT NULL,
    checkpoint_ns VARCHAR2(64) NOT NULL,
    channel VARCHAR2(128) NOT NULL,
    version VARCHAR2(128) NOT NULL,
    type VARCHAR2(64) NOT NULL,
    blob BLOB,
    CONSTRAINT pk_checkpoint_blobs PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
)""",
    # support empty blob
    """CREATE TABLE checkpoint_writes (
    thread_id VARCHAR2(128) NOT NULL,
    checkpoint_ns VARCHAR2(64) NOT NULL,
    checkpoint_id VARCHAR2(64) NOT NULL,
    task_id VARCHAR2(64) NOT NULL,
    idx NUMBER NOT NULL,
    channel VARCHAR2(128) NOT NULL,
    type VARCHAR2(64),
    blob BLOB,
    task_path VARCHAR2(128) NOT NULL,
    CONSTRAINT pk_checkpoint_writes PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
)""",
    # This is the table only for join checkpoint_blobs
    """CREATE TABLE checkpoint_channel_versions (
    thread_id VARCHAR2(128) NOT NULL,
    checkpoint_ns VARCHAR2(64) NOT NULL,
    checkpoint_id VARCHAR2(64) NOT NULL,
    channel VARCHAR2(128) NOT NULL,
    version VARCHAR2(128) NOT NULL,
    CONSTRAINT pk_checkpoint_channel_versions PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel)
)""",
    """CREATE INDEX idx_checkpoints_thread_id ON checkpoints(thread_id)""",
    """CREATE INDEX idx_checkpoint_blobs_thread_id ON checkpoint_blobs(thread_id)""",
    """CREATE INDEX idx_checkpoint_writes_thread_id ON checkpoint_writes(thread_id)""",
    """CREATE INDEX idx_checkpoint_channel_versions ON checkpoint_channel_versions(thread_id)""",
]

SELECT_SQL = f"""
SELECT 
    c.thread_id,
    c.checkpoint,
    CASE 
        WHEN c.checkpoint_ns = '__|default|__' THEN NULL
        ELSE c.checkpoint_ns
    END as checkpoint_ns,
    c.checkpoint_id,
    c.parent_checkpoint_id,
    c.metadata, 
    (
        SELECT JSON_ARRAYAGG(
            JSON_ARRAY(channel, version)
            RETURNING CLOB
        )
        FROM (
            SELECT
                cv.channel,
                cv.version
            FROM checkpoint_channel_versions cv
            WHERE cv.thread_id = c.thread_id
            AND cv.checkpoint_ns = c.checkpoint_ns
            AND cv.checkpoint_id = c.checkpoint_id
        )
    ) AS channel_versions,
    (
        SELECT JSON_ARRAYAGG(
            JSON_ARRAY(channel, type, blob RETURNING CLOB)
            RETURNING CLOB
        )
        FROM (
            SELECT
                cv.channel,
                bl.type,
                bl.blob
            FROM (
                SELECT
                    cvi.channel,
                    cvi.version
                FROM checkpoint_channel_versions cvi
                WHERE cvi.thread_id = c.thread_id
                AND cvi.checkpoint_ns = c.checkpoint_ns
                AND cvi.checkpoint_id = c.checkpoint_id
            ) cv
            JOIN checkpoint_blobs bl
              ON bl.thread_id = c.thread_id
             AND bl.checkpoint_ns = c.checkpoint_ns
             AND bl.channel = cv.channel
             AND bl.version = cv.version
        )
    ) AS channel_values,
 
    (
        SELECT JSON_ARRAYAGG(
            JSON_ARRAY(task_id, channel, type, blob RETURNING CLOB)
            RETURNING CLOB
        )
        FROM (
            SELECT
                cw.task_id,
                cw.channel,
                cw.type,
                cw.blob
            FROM checkpoint_writes cw
            WHERE cw.thread_id = c.thread_id
              AND cw.checkpoint_ns = c.checkpoint_ns
              AND cw.checkpoint_id = c.checkpoint_id
            ORDER BY cw.task_id, cw.idx
        )
    ) AS pending_writes,

    (
        SELECT JSON_ARRAYAGG(
            JSON_ARRAY(type, blob RETURNING CLOB)
            RETURNING CLOB
        )
        FROM (
            SELECT
                cw.type,
                cw.blob
            FROM checkpoint_writes cw
            WHERE cw.thread_id = c.thread_id
              AND cw.checkpoint_ns = c.checkpoint_ns
              AND cw.checkpoint_id = c.parent_checkpoint_id
              AND cw.channel = '{TASKS}'
            ORDER BY cw.task_path, cw.task_id, cw.idx
        )
    ) AS pending_sends

FROM checkpoints c
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
MERGE INTO checkpoint_blobs target
USING (SELECT :1 as thread_id, NVL(:2, '__|default|__') as checkpoint_ns, :3 as channel, :4 as version, :5 as type, :6 as blob FROM dual) source
ON (target.thread_id = source.thread_id AND target.checkpoint_ns = source.checkpoint_ns 
    AND target.channel = source.channel AND target.version = source.version)
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (source.thread_id, source.checkpoint_ns, source.channel, source.version, source.type, source.blob)
"""

UPSERT_CHECKPOINTS_SQL = """
MERGE INTO checkpoints target
USING (SELECT :1 as thread_id, NVL(:2, '__|default|__') as checkpoint_ns, :3 as checkpoint_id, :4 as parent_checkpoint_id, 
              :5 as checkpoint, :6 as metadata FROM dual) source
ON (target.thread_id = source.thread_id AND target.checkpoint_ns = source.checkpoint_ns 
    AND target.checkpoint_id = source.checkpoint_id)
WHEN MATCHED THEN
    UPDATE SET 
        checkpoint = source.checkpoint,
        metadata = source.metadata
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id, source.parent_checkpoint_id, 
            source.checkpoint, source.metadata)
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
MERGE INTO checkpoint_writes target
USING (SELECT :1 as thread_id, NVL(:2, '__|default|__') as checkpoint_ns, :3 as checkpoint_id, :4 as task_id, :5 as task_path,
              :6 as idx, :7 as channel, :8 as type, :9 as blob FROM dual) source
ON (target.thread_id = source.thread_id AND target.checkpoint_ns = source.checkpoint_ns 
    AND target.checkpoint_id = source.checkpoint_id AND target.task_id = source.task_id 
    AND target.idx = source.idx)
WHEN MATCHED THEN
    UPDATE SET 
        channel = source.channel,
        type = source.type,
        blob = source.blob
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id, source.task_id, source.task_path,
            source.idx, source.channel, source.type, source.blob)
"""
UPSERT_CHECKPOINT_CHANNEL_VERSIONS_SQL = """
MERGE INTO checkpoint_channel_versions target
USING (SELECT :1 as thread_id, NVL(:2, '__|default|__') as checkpoint_ns, :3 as checkpoint_id, :4 as channel, :5 as version FROM dual) source
ON (target.thread_id = source.thread_id AND target.checkpoint_ns = source.checkpoint_ns
    AND target.checkpoint_id = source.checkpoint_id AND target.channel = source.channel)
WHEN MATCHED THEN
    UPDATE SET
        version = source.version
WHEN NOT MATCHED THEN
    INSERT (thread_id, checkpoint_ns, checkpoint_id, channel, version)
    VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint_id, source.channel, source.version)
"""

INSERT_CHECKPOINT_WRITES_SQL = """
INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
VALUES (:1, NVL(:2, '__|default|__'), :3, :4, :5, :6, :7, :8, :9)
"""


class BaseOracleSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_CHANNEL_VERSIONS_SQL = UPSERT_CHECKPOINT_CHANNEL_VERSIONS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    jsonplus_serde = JsonPlusSerializer()
    supports_pipeline: bool

    def _load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        channel_values: Optional[list[tuple[str, str, bytes]]],
        pending_sends: Optional[list[tuple[str, bytes]]],
        channel_versions: Optional[list[tuple[str, str]]],
    ) -> Checkpoint:
        return {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed((send[0], send[1]))
                for send in pending_sends or []
            ],
            "channel_values": self._load_blobs(channel_values),
            "channel_versions": {k: v for k, v in channel_versions or []},
        }

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        return {**checkpoint, "pending_sends": []}

    def _load_blobs(
        self, blob_values: Optional[list[tuple[str, str, bytes]]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k: self.serde.loads_typed((t, v)) for k, t, v in blob_values if t != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, Optional[bytes]]]:
        if not versions:
            return []
        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values and values[k] is not None
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: Optional[list[tuple[str, str, str, bytes]]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid,
                    channel,
                    self.serde.loads_typed((t, v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _load_metadata(self, metadata: dict[str, Any]) -> CheckpointMetadata:
        return self.jsonplus_serde.loads(self.jsonplus_serde.dumps(metadata))

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        # NOTE: we're using JSON serializer (not msgpack), so we need to remove null characters before writing
        return serialized_metadata.decode().replace("\\u0000", "")

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        if current is None:
            return "0"
        return str(int(current) + 1)

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: MetadataInput,
        before: Optional[RunnableConfig] = None,
    ) -> tuple[str, list[Any]]:
        where_clauses = []
        args = []
        if config:
            where_clauses.append("c.thread_id = :1")
            args.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                where_clauses.append(
                    f"c.checkpoint_ns = NVL(:{len(args) + 1}, '__|default|__')"
                )
                args.append(checkpoint_ns)
            if checkpoint_id := get_checkpoint_id(config):
                where_clauses.append(f"c.checkpoint_id = :{len(args) + 1}")
                args.append(checkpoint_id)
        if filter:

            def flatten_dict(
                prefix: str, d: dict[str, Any]
            ) -> Generator[tuple[str, Any], None, None]:
                for key, value in d.items():
                    full_key = f"{prefix}.{key}"
                    if isinstance(value, dict):
                        yield from flatten_dict(full_key, value)
                    else:
                        yield (
                            f"JSON_VALUE(c.metadata, '{full_key}') = :{len(args) + 1}",
                            value,
                        )

            for clause, value in flatten_dict("$", filter):
                where_clauses.append(clause)
                args.append(value)
        if before:
            where_clauses.append("c.checkpoint_id < :" + str(len(args) + 1))
            args.append(get_checkpoint_id(before))
        if where_clauses:
            return " WHERE " + " AND ".join(where_clauses), args
        return "", []
