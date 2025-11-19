import datetime
import random

from langgraph.checkpoint.oracle import OracleSaver
from langgraph.graph import START, MessagesState, StateGraph
from oracledb import ConnectParams

def node1(state: MessagesState):
    print("Calling model with state:", len(state['messages']))
    response = 'msg1-' + str(random.randint(0,100))
    return {'messages': response}

def node2(state: MessagesState):
    print("Calling model with state:", len(state['messages']))
    response = 'msg2-' + str(random.randint(0,100))
    return {'messages': response}

builder = StateGraph(MessagesState)
builder.add_node("node2", node2)
builder.add_node("node1", node1)
builder.add_edge("node1", "node2")
builder.add_edge(START, "node1")
config = {"configurable": {"thread_id": datetime.datetime.now().isoformat()}}

params = ConnectParams(
    host="localhost",
    port=1521,
    user="system",
    password="test",
)
with OracleSaver.from_conn_params(params) as checkpointer:
    # setup() migrate DB.
    # If you created the table manually, you don't have to call setup().
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)
    res = graph.invoke(
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'I am user A',
                }
            ]
        },
        config,
    )

    checkpoints = list(checkpointer.list(config))
    for chk in checkpoints:
        print(chk)
    first_channel_values = checkpoints[0].checkpoint['channel_values']
    assert first_channel_values == res
