from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler


def load_agent() -> AgentExecutor:
    llm = ChatOpenAI(temperture=0.7, streaming=True)
    tools = load_tools(tool_name=["ddg-search", "wolfram-alpha", "arxiv", "wikipedia"], llm=llm)
    return initialize_agent(
        llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


chain = load_agent()
st_callback = StreamlitCallbackHandler(st.container())
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.run(prompt, callbacks=[st_callback])
        st.write(response)

