import os
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai.llms import OpenAI

openai_key = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(temperature=0.)

python_repl = PythonREPL()
tool = Tool(name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,)


agent = initialize_agent(tools=[tool, ], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.invoke('What is 872982 * 2391829')