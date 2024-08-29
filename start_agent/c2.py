from langchain import hub
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent  # TODO
from settings import OPENAI_API_KEY, SERPAPI_API_KEY, LANGSMITH_KEY


prompt = hub.pull(owner_repo_commit='hwchase17/react', api_key=LANGSMITH_KEY)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
searcher = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(name='Search',
             func=searcher.run,
             description='当大模型没有相关知识时，用于搜索知识'
             )
]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": '当前Agent最新研究进展是什么？'})

if __name__ == '__main__':
    agent_executor.invoke({"input": '当前Agent最新研究进展是什么？'})
    # print(prompt)
