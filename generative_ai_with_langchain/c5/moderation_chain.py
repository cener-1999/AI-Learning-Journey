from langchain.prompts import PromptTemplate
from langchain_openai.llms import OpenAI
from langchain.chains.moderation import OpenAIModerationChain
from langchain.schema import StrOutputParser
from pathlib import Path
import sys
cur_path = Path.cwd()
root_path = cur_path.parent.parent.parent
sys.path.append(str(root_path))
from settings import set_environment
set_environment()

cot_prompt = PromptTemplate.from_template("{question} \nLet's think step by step!")
llm_chain = cot_prompt | OpenAI() | StrOutputParser()
moderation_chain = OpenAIModerationChain()

chain = llm_chain|moderation_chain
r = chain.invoke({"question": "What is the future of programming?"})
print(r)