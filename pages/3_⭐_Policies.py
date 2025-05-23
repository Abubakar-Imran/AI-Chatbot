import utils
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Policies", page_icon="📄")
st.header('Policies')

class PyramidPolicyChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()
        self.page_key = "policy_messages"
        self.qa_chain = self.setup_qa_chain()

    @st.spinner('Generating Response...')
    def setup_qa_chain(self):
        docs = []
    
        loader = PyPDFLoader('./policies/SP_Policies.pdf')
        docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        system_prompt = (
            "You are an assistant for answering policies details of company (SoftPyramid). "
            "Use the following pieces of retrieved context of policies to answer "
            "the question. Give exact policy details."
            "If you don't know the answer, ask the user to enter more detail."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    @utils.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            utils.display_msg(user_query, 'user', self.page_key)

            response = self.qa_chain.invoke({"input": user_query})
            utils.display_msg(response["answer"], 'assistant', self.page_key)

            utils.print_qa(PyramidPolicyChatbot, user_query, response)

if __name__ == "__main__":
    obj = PyramidPolicyChatbot()
    obj.main()
