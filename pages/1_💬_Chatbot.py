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
        self.page_key = "chatbot_messages" 
    
    def setup_chain(self):
        prompt = ChatPromptTemplate.from_template("{query}")
        chain = prompt | self.llm | StrOutputParser()
        return chain
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            utils.display_msg(user_query, 'user', self.page_key)
            response = chain.invoke({"query": user_query})
            utils.display_msg(response, 'assistant', self.page_key)
            utils.print_qa(PyramidChatbot, user_query, response)

if __name__ == "__main__":
    chatbot = PyramidChatbot()
    chatbot.main()