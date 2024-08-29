"""
1. use fake_llm for rapid prototyping and unit testing agents
2. use build-in tools
3. use agent
"""

from langchain_community.llms.fake import FakeListLLM
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

fake_llm = FakeListLLM(responses=['Hello'])
print(fake_llm)

python_repl = PythonREPL()
tool = Tool(name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,)

responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
fake_llm = FakeListLLM(responses=responses)

agent = initialize_agent(tools=[tool, ], llm=fake_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run('What is 2+2?')

