from transformers import pipeline
import torch
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

generate_text = pipeline(model="aisquared/dlite-v1-355m",
                         torch_dtype=torch.bfloat16,
                         trust_remote_code=True,
                         device_map="auto",
                         framework="pt")
llm = generate_text("In this chapter, we'll discuss first steps with generative AI in Python.")
print(llm)

prompt = PromptTemplate(input_variables=['question'], template='Q: {question} A: let us think step by step')
llm_chain = LLMChain(llm=llm, prompt=prompt)
question = "What is electroencephalography?"
print(llm_chain.run(question))
