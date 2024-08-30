import streamlit as st
from streamlit.logger import get_logger
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger('Langchain-Chatbot')

def enable_chat_history(func):
    def wrapper(*args, **kwargs):
        instance = args[0]
        page_key = instance.page_key

        if page_key not in st.session_state:
            st.session_state[page_key] = []
            st.session_state[page_key].append({"role": "assistant", "content": "How can I help you?"})

        for msg in st.session_state[page_key]:
            st.chat_message(msg["role"]).write(msg["content"])

        func(*args, **kwargs)

    return wrapper

def display_msg(msg, author, page_key):
    st.session_state[page_key].append({"role": author, "content": msg})
    st.chat_message(author).write(msg)
    
def configure_llm():
    llm = ChatOllama(model="llama3", base_url=st.secrets["OLLAMA_ENDPOINT"])
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
