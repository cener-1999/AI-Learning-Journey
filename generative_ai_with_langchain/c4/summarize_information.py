from langchain_community.llms import OpenAI
from langchain_decorators import llm_prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback

"""
=== Prompt ===
"""
# 简单的chain
prompt = """
Summarize this text in one sentence

{text}
"""

text = """"""
llm = OpenAI()
summary = llm(prompt.format(text=text))


# 使用Langchain装饰器
@llm_prompt
def summarize(text: str, length="short") -> str:
    """
    Summarize this text in {length} length:  {text}
    """
    return


summary = summarize(text="let me tell you a boring story from when I was young...")

# 使用PromptTemplate
llm = OpenAI()
prompt = PromptTemplate.from_template("Summarize this text: {text}?"  )
runnable = prompt | llm | StrOutputParser()
summary = runnable.invoke({"text": text})


"""
=== COT ===

Incrementally increase the information density of GPT-4 generated summaries  while controlling length.
"""

template = """
    Article: { text }  
    You will generate increasingly concise, entity-dense summaries of the above article.  
    Repeat the following 2 steps 5 times.  
    Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.  
    Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.
            A missing entity is:  
            - relevant to the main story,  
            - specific yet concise (5 words or fewer),  
            - novel (not in the previous summary),  
            - faithful (present in the article),  
            - anywhere (can be located anywhere in the article).
    Guidelines:  
    - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little 
      information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article 
      discusses") to reach ~80 words. 
    - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.  
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses". 
    - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article. 
    - Missing entities can appear anywhere in the new summary.  
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.  
    Remember, use the exact same number of words for each summary.  
    Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
    """

"""
=== Map-Reduce pipelines ===

The default prompt for both the map and reduce steps is this:
    'Write a concise summary of the following:  {text}  CONCISE SUMMARY:'
"""
pdf_file_path = ''
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

llm = OpenAI()
chain = load_summarize_chain(chain_type='map_reduce', llm=llm)
chain.run()


"""
=== Monitoring token usage ===
"""
llm_chain = PromptTemplate.from_template("Tell me a joke about {topic}!") | OpenAI()
with get_openai_callback() as cb:
    response = llm_chain.invoke(dict(topic="light bulbs"))
    print(response)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")


