# LangGraph Checkpoint Oracle

Implementation of LangGraph CheckpointSaver that uses Oracle Database.

> [!TIP]
> The code in this repository tries to mimic the code in [langgraph-checkpoint-postgres](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres) as much as possible to enable keeping in sync with the official checkpointer implementation.

> [!NOTE]
> This implementation uses Oracle-specific SQL syntax and data types. Oracle Database 19c or later is recommended.

## Supported Oracle Database
Oracle Database 18c or later is recommended.

## Dependencies

To use Both synchronous `OracleSaver` and asynchronous `AsyncOracleSaver`, install the `oracledb` package.

There is currently no support for other drivers.

## Usage

> [!IMPORTANT]
> When using Oracle checkpointers for the first time, make sure to call the `.setup()` method to create required tables. See example below.

> [!IMPORTANT]
> When manually creating Oracle connections and passing them to `OracleSaver`, pay attention to transaction management.

> [!IMPORTANT]
`__|default|__` is not allowed as a `checkpoint_ns` because it is reserved for internal use.

```python
from oracledb import ConnectParams
from langgraph.checkpoint.oracle import OracleSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

DB_PARAMS = {
    "user": "user",
    "password": "password",
    "dsn": "localhost:1521/ORCLPDB1",
}
params = ConnectParams(host="localhost", port=1521, service_name="FREE", user="system", password="test")
with OracleSaver.from_conn_params(DB_PARAMS) as checkpointer:
    # call .setup() the first time you're using the checkpointer
    checkpointer.setup()
    checkpoint = {
        "v": 2,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```

### Async

```python
from oracledb import ConnectParams
from langgraph.checkpoint.oracle.aio import AsyncOracleSaver

params = ConnectParams(host="localhost", port=1521, service_name="FREE", user="system", password="test")
async with AsyncOracleSaver.from_conn_params(params) as checkpointer:
    checkpoint = {
        "v": 2,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # store checkpoint
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # load checkpoint
    await checkpointer.aget(read_config)

    # list checkpoints
    [c async for c in checkpointer.alist(read_config)]
```
