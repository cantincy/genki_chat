import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from prompt import SYSTEM_PROMPT

st.set_page_config(
    page_title="元气AI小樱",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)


class Agent:
    def __init__(self, persist_dir: str = "./chroma_db", temperature: float = 0):
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings()
        )

        retriever = self.vector_store.as_retriever()

        memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="chat_history",
        )

        self.llm = ChatOpenAI(
            temperature=temperature,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        self.agent = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

    def invoke(self, query: str) -> str:
        try:
            res = self.agent.invoke({"input": query})
            return res.get("text", "抱歉，我遇到了一些问题。")
        except Exception as e:
            st.error(f"Error: {e}")
            return "抱歉，服务暂时不可用，请稍后再试。"


@st.cache_resource(ttl=3600)
def create_agent() -> Agent:
    return Agent(persist_dir="./chroma_db", temperature=0.8)


def main():
    st.title("🌸 元气AI小樱")
    st.caption("随时为您解答问题的智能助手～")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    else:
        for message in st.session_state['chat_history']:
            if message["role"] == 'user':
                st.chat_message('user').text(message["content"])
            else:
                st.chat_message('assistant').text(message["content"])

    if user_query := st.chat_input("请输入你的问题："):
        st.chat_message('user').text(user_query)
        st.session_state['chat_history'].append(
            {"role": "user", "content": user_query})

        with st.spinner("小樱正在思考..."):
            try:
                agent = create_agent()
                response = agent.invoke(user_query)
            except Exception as e:
                response = f"系统错误: {str(e)}"
        print(response)

        st.chat_message('assistant').text(response)
        st.session_state['chat_history'].append(
            {"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
