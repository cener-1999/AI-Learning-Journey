"""
not support for openai >= 1.0.0
run `pip install openai==0.
"""
from langchain_community.chat_models import JinaChat
from langchain.schema import HumanMessage

chat = JinaChat(temperature=0)
message = [
    HumanMessage(
        content='把这句话翻译成英文: 我喜欢生成式人工智能'
    )
]

print(chat(message))
