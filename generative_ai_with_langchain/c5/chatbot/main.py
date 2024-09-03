import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from document_loader import DocumentLoader
from chains import chain, read_doc_init_retriever

from pathlib import Path
import sys
cur_path = Path.cwd()
root_path = cur_path.parent.parent.parent
sys.path.append(str(root_path))
from settings import set_environment
set_environment()


def main():
    st.set_page_config(page_title='Hello Joker :-)', page_icon='🤡')
    st.title("🤡 小丑文本对话")

    uploaded_files = st.file_uploader(
        label='上传文件',
        type=list(DocumentLoader.supported_extensions.keys()),
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("请上传文件~")
        st.stop()

    retriever = read_doc_init_retriever(uploaded_files)
    qa_chain = chain(retriever)
    assistant = st.chat_message("Joker King")
    user_query = st.chat_input(placeholder="你好, 小丑 🤡")

    if user_query:
        stream_handler = StreamlitCallbackHandler(assistant)
        response = qa_chain.invoke({'input': user_query}, config={'callbacks': [stream_handler]})
        st.markdown(response.get('answer'))


if __name__ == '__main__':
    # streamlit run ./main.py
    main()



