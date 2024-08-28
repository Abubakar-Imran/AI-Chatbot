import streamlit as st

st.set_page_config(
    page_title="Pyramid AI Chatbot",
    page_icon='💬',
    layout='wide'
)
# st.image('./assets/pyramid-ai.png')
st.header("Pyramid AI Chatbot Implementations using Langchain")
st.write("""
[![website ](https://img.shields.io/badge/SoftPyramid.com-gray)](https://softpyramid.com/)
[![linkedin ](https://img.shields.io/badge/SoftPyramid-gray?logo=linkedin)](https://www.linkedin.com/company/softpyramid/)
""")
# ![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Flangchain-chatbot.streamlit.app&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
st.write("""
Langchain is a powerful framework designed to streamline the development of applications using Language Models (LLMs). It provides a comprehensive integration of various components, simplifying the process of assembling them to create robust applications.

Leveraging the power of Langchain, the creation of chatbots becomes effortless. Here are a few examples of chatbot implementations catering to different use cases:

- **Basic Chatbot**: Engage in interactive conversations with the LLM.
- **Context aware chatbot**: A chatbot that remembers previous conversations and provides responses accordingly.
- **Chatbot with Internet Access**: An internet-enabled chatbot capable of answering user queries about recent events.
- **Chat with your documents**: Empower the chatbot with the ability to access custom documents, enabling it to provide answers to user queries based on the referenced information.
- **Chat with SQL database**: Enable the chatbot to interact with a SQL database through simple, conversational commands.
- **Chat with Websites**: Enable the chatbot to interact with website contents.

To explore sample usage of each chatbot, please navigate to the corresponding chatbot section.
""")