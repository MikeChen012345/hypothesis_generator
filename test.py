import os
import dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver  

dotenv.load_dotenv()

model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.5,
    timeout=10,
    max_retries=3,
    max_tokens=4096,
    base_url=os.getenv("OPENAI_API_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

DB_URI = os.getenv("POSTGRESQL_CONNECTION_STRING")
assert DB_URI is not None, "POSTGRESQL_CONNECTION_STRING must be set in .env file"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # Create necessary tables if they don't exist. No side effects if they do.

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)  

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,  
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()