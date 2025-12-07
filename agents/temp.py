import os
from pprint import pprint
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

### Configuration
load_dotenv()

config = {"configurable": {"session_id": "test-session"}}


### Initialize chat model
model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.5,
    timeout=10,
    max_retries=3,
    max_tokens=4096,
    base_url=os.getenv("OPENAI_API_ENDPOINT"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

system_prompt = """You are an advanced AI assistant designed to help users with a variety of tasks.
You can understand and respond to complex queries, provide detailed explanations, and assist with problem-solving."""

agent = create_agent(
    model=model,
    tools=[],
    system_prompt=system_prompt,
)

memories = {}
user_input = ""

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in memories:
        memories[session_id] = ChatMessageHistory()
    return memories[session_id]

qa_prompt = ChatPromptTemplate.from_messages([
    # ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{user_input}"),
])

chain = qa_prompt | agent
chain_with_memory = RunnableWithMessageHistory(runnable=chain, 
                        get_session_history=get_session_history,
                        input_messages_key="user_input", history_messages_key="chat_history")


# Render the conversation history
while True:
    user_input = input("\n")
    response = chain_with_memory.invoke(
        {
            "user_input": user_input
        },
        config=config
    )
    print("==================")
    print(memories)
    print(get_session_history(config["configurable"]["session_id"]))

