from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    temperature=0.8,
    model_kwargs={'max_length': 512},
    top_k=50,
    repo_id="google/flan-t5-xxl",
)

prompt = 'Beijing in which country?'
completion = llm.invoke(prompt)
