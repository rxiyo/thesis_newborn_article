from langchain_core.messages import HumanMessage, SystemMessage
from tools.poe_chat_client import PoeChatOpenAI

chat = PoeChatOpenAI(model="gpt-5")  # 模型名称请确认 Poe 支持
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!"),
]

try:
    response = chat.batch([messages])
    print("Raw response object:", response)
    print("Extracted content:", response[0].content)
except Exception as exc:
    print("Caught exception:", exc)