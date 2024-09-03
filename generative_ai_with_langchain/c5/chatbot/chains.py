from langchain.chains.base import Chain
from langchain.schema import BaseRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os
import tempfile
from document_loader import load_document
from vector_storage import configure_retriever


def chain(retriever: BaseRetriever) -> Chain:
    """
    Configure chain with a retriever
    usage: chain.invoke({"input": "", "chat_history": [],})
    """

    # TODO 缺少获取的chat_history的实现
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # set temperature low to keep hallucinations in check
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    # 结合从history_chat中获取上下文的retriever
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_doc_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 链接上 retriever
    convo_qa_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_doc_chain)
    return convo_qa_chain


def read_doc_init_retriever(uploaded_files) -> BaseRetriever:
    """Read documents, configure retriever, and the chain."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs)
    return retriever


if __name__ == '__main__':
    docs = load_document('../tmp/short.txt')
    retriever = configure_retriever(docs=docs)
    qa_chain = chain(retriever=retriever)
    res = qa_chain.invoke({'input': 'Hi'}, config={'callbacks': []})
    print(res)
