import os
from types import SimpleNamespace
from typing import Iterable, List

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class PoeChatOpenAI:
    """
    兼容 LangChain ChatOpenAI 的简易包装：
    - 支持 batch([[BaseMessage,...], ...])
    - 默认读取环境变量中的 POE API Key 和 Base URL
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, temperature: float | None = None, **kwargs):
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("POE_API_KEY"),
            base_url="https://api.poe.com/v1",
        )

    def batch(self, batch_messages: Iterable[List[BaseMessage]]) -> List[SimpleNamespace]:
        responses = []
        for conversation in batch_messages:
            payload = [self._convert_message(msg) for msg in conversation]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=payload,
            )
            # print("Completion:", completion)  # 调试用
            responses.append(SimpleNamespace(content=completion.choices[0].message.content))
        return responses

    @staticmethod
    def _convert_message(message: BaseMessage) -> dict:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")
        return {"role": role, "content": message.content}