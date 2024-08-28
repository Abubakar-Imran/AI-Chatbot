import utils
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')
st.write('Allows users to interact with the LLM')

class PyramidChatbot:
    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
    
    def setup_chain(self):
        prompt = ChatPromptTemplate.from_template(" {query}")
        chain = prompt | self.llm | StrOutputParser()
        return chain
    
    def main(self):
        chain = self.setup_chain()
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
                
        user_query = st.chat_input(placeholder="Ask me anything!")
        
        if user_query:
            utils.display_msg(user_query, 'user')
            chain = self.setup_chain()
            response = chain.invoke({"query": user_query})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            utils.print_qa(PyramidChatbot, user_query, response)
            
if __name__ == "__main__":
    chatbot = PyramidChatbot()
    chatbot.main()