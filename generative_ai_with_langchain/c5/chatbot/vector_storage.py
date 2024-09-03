from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from typing import List


def configure_retriever(docs: List[Document], use_compression=True) -> BaseRetriever:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(documents=splits, embedding=embedding)
    # maximum marginal relevance
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={"k": 2, "fetch_k": 4})

    if not use_compression:
        return retriever

    embedding_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.76)
    retriever = ContextualCompressionRetriever(base_compressor=embedding_filter,
                                               base_retriever=retriever)
    return retriever


if __name__ == '__main__':
    from document_loader import load_document
    docs = load_document('../tmp/langchain.txt')
    configure_retriever(docs)
