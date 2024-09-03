import utils
import requests
import streamlit as st
from flask import Flask, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Attendence Chatbot", page_icon="📆")
st.header('Attendence Chatbot')

class PyramidAttendenceChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.page_key = "attendence_messages" 
    
    def setup_chain(self):
        prompt = ChatPromptTemplate.from_template("You are a helpful attendence assistant and chatbot for SoftPyramid Company. You reply with efficient answers. Ask for additional information if you are not confident about the answer.\n {query}")
        chain = prompt | self.llm | StrOutputParser()
        return chain
    
    def check_for_attendance(self, user_query, chain):
        input = f"""
        You are tasked with determining the intent of the user input.
        Question: Does the following input ask for an attendance record and not for marking a leave? Your response must be either 'yes' or 'no' and nothing else. Do not provide any additional explanations or information.
        User Input: "{user_query}"
        """
        response = chain.invoke({"query": input})
        return response.lower()

    def check_for_leave(self, user_query, chain):
        input = f"""
        You are tasked with determining the intent of the user input.
        Question: Does the following input ask for marking a leave? Your response must be either 'yes' or 'no' and nothing else. Do not provide any additional explanations or information.
        User Input: "{user_query}"
        """
        response = chain.invoke({"query": input})
        return response.lower()

    def check_is_input_in_leave_format(self, user_query, chain):
        input = f"""
        You are tasked with determining if the user input matches the leave format.
        Question: Is the following input in the format (dd-mm-yyyy, leave_type, leave_category)?
        Where Leave type must be from [Work From Home, Hour Break, Short Break, Half day, Full day] and Leave category must be from [Sick Leave, Casual Leave, Annual Leave, Bereavement Leave].
        Your response must be either 'yes' or 'no' and nothing else. Do not provide any additional explanations or information.
        User Input: "{user_query}"
        """
        response = chain.invoke({"query": input})
        return response.lower()
    
    def process_attendance_report(self):
        url = st.secrets["ATTENDENCE_RECORD_ENDPOINT"]  
        response = requests.get(url)
        return response.json()

    def mark_leave(self, date, leave_category, leave_type):
        leave_url = st.secrets["MARK_LEAVE_ENDPOINT"]  
        leave_data = {'date': date, 'leave_type': leave_type, 'leave_category': leave_category}
        response = requests.post(leave_url, json=leave_data)
        return response.json()
    
    @utils.enable_chat_history
    def main(self):
                
        app = Flask(__name__)
        chain = self.setup_chain()
        
        @app.route('/report', methods=['GET'])
        def report():
            response = self.process_attendance_report()
            return jsonify(response)

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            prompt = ""
            for message in st.session_state[self.page_key]:
                prompt += f"{message['role']}: {message['content']}\n" 
            
            utils.display_msg(user_query, 'user', self.page_key)
            
            is_mark_leave_input = self.check_for_leave(user_query, chain)
            is_attendance_record_input = self.check_for_attendance(user_query, chain)
            leave_input = self.check_is_input_in_leave_format(user_query, chain)
            
            print(is_attendance_record_input, is_mark_leave_input, leave_input)

            response = ""
            
            if is_attendance_record_input == "yes":
                attendance_report = self.process_attendance_report()
                prompt += f"system: This is the attendance record of this user.\n {str(attendance_report)}\n show output in table format. If data not avalaible donot show sample data. just tell user that data is not avalaible"
                response = chain.stream({"query": prompt})
                utils.display_msg(response, 'assistant', self.page_key)
            elif is_mark_leave_input == "yes":
                if leave_input == "yes":
                    try:
                        date, leave_type, leave_category = map(str.strip, user_query.split(","))
                        response = self.mark_leave(date, leave_category, leave_type)
                        feedback = f"Leave marked for {date}. Leave Category: {leave_category}, Leave Type: {leave_type}."
                        utils.display_msg(feedback, 'assistant', self.page_key)
                    except ValueError:
                        feedback = f"""
                        Please provide the leave details in the format (dd-mm-yyyy, leave type, leave category).\n
                        Leave type : [Work From Home, Hour Break, Short Break, Half day, Full day]\n
                        Leave category : [Sick Leave, Casual Leave, Annual Leave, Bereavement Leave]\n
                        """
                        utils.display_msg(feedback, 'assistant', self.page_key)
                else:
                    leave_format = f"""
                    Please provide the leave details in the format (dd-mm-yyyy, leave type, leave category).\n
                    Leave type : [Work From Home, Hour Break, Short Break, Half day, Full day]\n
                    Leave category : [Sick Leave, Casual Leave, Annual Leave, Bereavement Leave].\n
                    """
                    utils.display_msg(leave_format, 'assistant', self.page_key)
            elif is_mark_leave_input == "no" and is_attendance_record_input == "no":
                response = chain.stream({"query": user_query})
                utils.display_msg(response, 'assistant', self.page_key)
            
            utils.print_qa(PyramidAttendenceChatbot, user_query, response)

if __name__ == "__main__":
    obj = PyramidAttendenceChatbot()
    obj.main()
