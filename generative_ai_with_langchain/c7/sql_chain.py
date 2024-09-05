from langchain_openai.llms import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from settings import OPENAI_API_KEY

db = SQLDatabase.from_uri("sqlite:///../../../../notebooks/Chinook.db")
llm = OpenAI(temperature=0, verbose=True, openai_api_key=OPENAI_API_KEY)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
db_chain.invoke({'input': "How many employees are there?"})

