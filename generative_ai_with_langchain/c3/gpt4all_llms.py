from langchain_community.llms import GPT4All

model = GPT4All(model="mistral-7b-openorca.Q4_0.gguf",
                n_ctx=512,
                n_threads=8)
response = model("We can run large language models locally for all kinds of applications, "  )