import threading
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Iterator, Literal, NamedTuple, Optional, TypeVar, Union, cast

import orjson
import oracledb  # cx_Oracleからoracledbに変更

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)

class Migration(NamedTuple):
    sql: str
    params: Optional[dict[str, Any]] = None
    condition: Optional[Callable[["BaseOracleStore"], bool]] = None

MIGRATIONS: Sequence[str] = [
    # テーブル作成
    """
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
    """,
    # インデックス作成
    """
    CREATE INDEX store_prefix_idx ON store (prefix)
    """,
    # expires_at index
    """
    CREATE INDEX idx_store_expires_at ON store (expires_at)
    """,
]

VECTOR_MIGRATIONS: Sequence[Migration] = [
    Migration(
        """
        CREATE TABLE store_vectors (
            prefix VARCHAR2(255) NOT NULL,
            key VARCHAR2(255) NOT NULL,
            field_name VARCHAR2(255) NOT NULL,
            embedding VECTOR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (prefix, key, field_name)
        )
        """,
    ),
    Migration(
        """
        CREATE INDEX store_vectors_embedding_idx ON store_vectors (embedding) INDEXTYPE IS VECTOR_INDEX
        """,
    ),
]

class OracleIndexConfig(IndexConfig, total=False):
    distance_type: Literal["euclidean", "cosine", "inner_product"]

class BaseOracleStore:
    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    conn: oracledb.Connection  # 型アノテーションをoracledb.Connectionに変更
    _deserializer: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]]
    index_config: Optional[OracleIndexConfig]

    def __init__(
        self,
        conn: oracledb.Connection,  # 引数の型も修正
        *,
        deserializer: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]] = None,
        index: Optional[OracleIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        self._deserializer = deserializer
        self.conn = conn
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._ttl_sweeper_thread: Optional[threading.Thread] = None
        self._ttl_stop_event = threading.Event()

    @contextmanager
    def _cursor(self) -> Iterator[oracledb.Cursor]:  # 型アノテーションを修正
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def setup(self) -> None:
        def _get_version(cur: oracledb.Cursor, table: str) -> int:  # 型アノテーションを修正
            try:
                cur.execute(f"CREATE TABLE {table} (v NUMBER PRIMARY KEY)")
            except oracledb.DatabaseError as e:  # 例外をoracledb.DatabaseErrorに変更
                if "ORA-00955" not in str(e):
                    raise
            cur.execute(f"SELECT v FROM {table} ORDER BY v DESC FETCH FIRST 1 ROWS ONLY")
            row = cur.fetchone()
            return -1 if row is None else row[0]

        with self._cursor() as cur:
            version = _get_version(cur, table="store_migrations")
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                try:
                    cur.execute(sql)
                    cur.execute("INSERT INTO store_migrations (v) VALUES (:1)", (v,))
                except oracledb.DatabaseError as e:
                    # ORA-00955: name is already used by an existing object
                    if "ORA-00955" in str(e):
                        cur.execute("INSERT INTO store_migrations (v) VALUES (:1)", (v,))
                        continue
                    logger.error(f"Failed to apply migration {v}.\nSql={sql}\nError={e}")
                    raise

            if self.index_config:
                version = _get_version(cur, table="vector_migrations")
                for v, migration in enumerate(self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1):
                    if migration.condition and not migration.condition(self):
                        continue
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        sql = sql % params
                    try:
                        cur.execute(sql)
                        cur.execute("INSERT INTO vector_migrations (v) VALUES (:1)", (v,))
                    except oracledb.DatabaseError as e:
                        if "ORA-00955" in str(e):
                            cur.execute("INSERT INTO vector_migrations (v) VALUES (:1)", (v,))
                            continue
                        logger.error(f"Failed to apply vector migration {v}.\nSql={sql}\nError={e}")
                        raise

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor() as cur:
            if GetOp in grouped_ops:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )

            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )
            if PutOp in grouped_ops:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, tuple, tuple[str, ...], list]]:
        """
        Build queries to fetch (and optionally refresh the TTL of) multiple keys per namespace for Oracle.
        Each returned element is a tuple of:
        (sql_query_string, sql_params, namespace, items_for_this_namespace, update_query, update_params)
        """
        namespace_groups = defaultdict(list)
        refresh_ttls = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(op.refresh_ttl)

        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items)
            this_refresh_ttls = refresh_ttls[namespace]

            update_query = None
            update_params = None
            if any(this_refresh_ttls):
                keys_to_refresh = [k for (_, k), r in zip(items, this_refresh_ttls) if r]
                if keys_to_refresh:
                    update_query = (
                        f"""
                        UPDATE store
                        SET expires_at = SYSTIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                        WHERE prefix = :1 AND key IN ({','.join([':{}'.format(i+2) for i in range(len(keys_to_refresh))])}) AND ttl_minutes IS NOT NULL
                        """
                    )
                    update_params = (namespace, *keys_to_refresh)

            select_query = (
                f"""
                SELECT key, value, created_at, updated_at
                FROM store
                WHERE prefix = :1 AND key IN ({','.join([':{}'.format(i+2) for i in range(len(keys))])})
                """
            )
            select_params = (namespace, *keys)

            results.append((select_query, select_params, namespace, items, update_query, update_params))

        return results

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        # Use _get_batch_GET_ops_queries to generate queries per namespace
        for select_query, select_params, namespace, items, update_query, update_params in self._get_batch_GET_ops_queries(get_ops):
            # Run TTL update if needed
            if update_query and update_params:
                cur.execute(update_query, update_params)
            # Fetch data
            cur.execute(select_query, select_params)
            rows = cur.fetchall()
            key_to_row = {row[0]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = Item(
                        key=row[0],
                        namespace=namespace,
                        value=orjson.loads(row[1].read() if hasattr(row[1], "read") else row[1]),
                        created_at=row[2],
                        updated_at=row[3],
                    )
                else:
                    results[idx] = None

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: oracledb.Cursor,
    ) -> None:
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        # 削除
        if deletes:
            namespace_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for op in deletes:
                namespace_groups[op.namespace].append(op.key)
            for namespace, keys in namespace_groups.items():
                cur.execute(
                    f"DELETE FROM store WHERE prefix = :1 AND key IN ({','.join([':{}'.format(i+2) for i in range(len(keys))])})",
                    (namespace, *keys),
                )
        # 挿入・更新
        if inserts:
            for op in inserts:
                value_json = orjson.dumps(op.value).decode("utf-8")
                if op.ttl is not None:
                    expires_at_expr = "SYSTIMESTAMP + NUMTODSINTERVAL(:5, 'SECOND')"
                    ttl_minutes = op.ttl
                else:
                    expires_at_expr = "NULL"
                    ttl_minutes = None
                # MERGEでUPSERT
                cur.execute(
                    f"""
                    MERGE INTO store s
                    USING (SELECT :1 AS prefix, :2 AS key FROM dual) src
                    ON (s.prefix = src.prefix AND s.key = src.key)
                    WHEN MATCHED THEN
                        UPDATE SET value = :3, updated_at = SYSTIMESTAMP, expires_at = {expires_at_expr}, ttl_minutes = :6
                    WHEN NOT MATCHED THEN
                        INSERT (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
                        VALUES (:1, :2, :3, SYSTIMESTAMP, SYSTIMESTAMP, {expires_at_expr}, :6)
                    """,
                    (op.namespace, op.key, value_json, None, op.ttl * 60 if op.ttl else None, ttl_minutes),
                )
                # ベクトル埋め込みも必要ならここで追加

    def _prepare_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> tuple[
        list[tuple[str, list]],
        list[tuple[int, str]],
    ]:
        """
        Build per-SearchOp SQL queries (with optional TTL refresh) plus embedding requests.
        Returns:
        - queries: list of (SQL, param_list)
        - embedding_requests: list of (original_index_in_search_ops, text_query)
        """
        queries = []
        embedding_requests = []
        for idx, (_, op) in enumerate(search_ops):
            filter_params = []
            filter_clauses = []
            if op.filter:
                for key, value in op.filter.items():
                    if isinstance(value, dict):
                        for op_name, val in value.items():
                            # Only $eq supported for now
                            if op_name == "$eq":
                                filter_clauses.append(f"JSON_VALUE(value, '$.{key}') = :{len(filter_params)+1}")
                                filter_params.append(val)
                            else:
                                raise ValueError(f"Unsupported operator: {op_name}")
                    else:
                        filter_clauses.append(f"JSON_VALUE(value, '$.{key}') = :{len(filter_params)+1}")
                        filter_params.append(value)

            ns_condition = "prefix = :ns"
            ns_param = op.namespace_prefix if op.namespace_prefix else op.namespace

            extra_filters = (
                " AND " + " AND ".join(filter_clauses) if filter_clauses else ""
            )

            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))
                # Oracle vector search SQL (simplified, no ANN index logic)
                # You may need to adjust for your Oracle vector index implementation
                vector_search_sql = f"""
                        SELECT prefix, key, value, created_at, updated_at,
                            VECTOR_DISTANCE(embedding, :vector) AS neg_score
                        FROM store_vectors
                        WHERE prefix = :ns
                        ORDER BY neg_score ASC
                        FETCH FIRST :limit ROWS ONLY
                    """

                search_results_sql = f"""
                        WITH scored AS (
                            {vector_search_sql}
                        )
                        SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at,
                            scored.neg_score AS score
                        FROM store s
                        JOIN scored ON s.prefix = scored.prefix AND s.key = scored.key
                        ORDER BY score DESC
                        OFFSET :offset ROWS FETCH NEXT :limit2 ROWS ONLY
                    """

                search_results_params = {
                    "vector": None,  # placeholder for embedding
                    "ns": ns_param,
                    "limit": op.limit * (self.index_config.get("__estimated_num_vectors", 1) * 2 + 1),
                    "offset": op.offset,
                    "limit2": op.limit,
                }
                # Oracle doesn't support positional placeholders in the same way as Postgres
                # so we use named parameters

            else:
                base_query = f"""
                        SELECT prefix, key, value, created_at, updated_at, NULL AS score
                        FROM store
                        WHERE {ns_condition}{extra_filters}
                        ORDER BY updated_at DESC
                        OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
                    """
                search_results_sql = base_query
                search_results_params = {
                    "ns": ns_param,
                    "offset": op.offset,
                    "limit": op.limit,
                }
                for i, val in enumerate(filter_params):
                    search_results_params[f"f{i}"] = val

            if op.refresh_ttl:
                # Oracle: update TTL for selected keys, then select
                final_sql = f"""
                        DECLARE
                            CURSOR c IS
                                {search_results_sql};
                        BEGIN
                            FOR r IN c LOOP
                                UPDATE store
                                SET expires_at = SYSTIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                                WHERE prefix = r.prefix AND key = r.key AND ttl_minutes IS NOT NULL;
                            END LOOP;
                        END;
                    """
                final_params = search_results_params.copy()
            else:
                final_sql = search_results_sql
                final_params = search_results_params

            queries.append((final_sql, final_params))

        return queries, embedding_requests

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        if embedding_requests and self.embeddings:
            vectors = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )
            for (idx, _), vector in zip(embedding_requests, vectors):
                params = queries[idx][1]
                if isinstance(params, dict):
                    params["vector"] = vector

        for (idx, _), (query, params) in zip(search_ops, queries):
            cur.execute(query, params)
            rows = cur.fetchall()
            results[idx] = [
                SearchItem(
                    value=orjson.loads(row[2].read() if hasattr(row[2], "read") else row[2]),
                    key=row[1],
                    namespace=row[0] if isinstance(row[0], tuple) else (row[0],),
                    created_at=row[3],
                    updated_at=row[4],
                    score=row[5] if len(row) > 5 else None,
                )
                for row in rows
            ]

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        for idx, op in list_ops:
            # Oracleで階層的なprefix取得
            cur.execute(
                """
                SELECT DISTINCT SUBSTR(prefix, 1, INSTR(prefix, '.', 1, :1)-1) AS truncated_prefix
                FROM store
                """,
                (op.max_depth,),
            )
            results[idx] = [row[0] for row in cur]

    def sweep_ttl(self) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < SYSTIMESTAMP
                """
            )
            return cur.rowcount

    def start_ttl_sweeper(self, sweep_interval_minutes: Optional[int] = None):
        if not self.ttl_config:
            return None
        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            return
        self._ttl_stop_event.clear()
        interval = float(sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5)
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        def _sweep_loop():
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception("Store TTL sweep iteration failed", exc_info=exc)
            except Exception as exc:
                logger.error("TTL sweeper thread failed", exc_info=exc)

        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()

    def stop_ttl_sweeper(self, timeout: Optional[float] = None) -> bool:
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True
        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()
        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")
        return success

    def __del__(self) -> None:
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)

# Utility function
def _ensure_index_config(index_config: OracleIndexConfig):
    # Same logic as Postgres
    index_config = index_config.copy()
    tokenized = []
    tot = 0
    text_fields = index_config.get("fields") or ["$"]
    if isinstance(text_fields, str):
        text_fields = [text_fields]
    for p in text_fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized
    index_config["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(index_config.get("embed"))
    return embeddings, index_config

def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot

