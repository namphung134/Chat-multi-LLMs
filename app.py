import google.generativeai as genai
import streamlit as st
import os
from dotenv import load_dotenv

from db import EasyMongo
from llm_strings import LLMStrings
from utils import output_text, simulate_response, create_message

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Đọc API key từ môi trường



if __name__ == '__main__':
    # Get LLM model
    llm = genai.GenerativeModel("gemini-2.0-flash")

    # App title
    st.title(LLMStrings.APP_TITLE)

    # Initial prompt
    with st.chat_message(LLMStrings.AI_ROLE):
        st.write(LLMStrings.GREETINGS)

    # Initialize chat history
    if LLMStrings.SESSION_STATES not in st.session_state:
        st.session_state.messages = []

    # Connect MongoDB
    mongo_server = EasyMongo()
    collection_name = mongo_server.get_collection()

    # Display chat messages from history on app rerun
    messages = collection_name.find()
    for message in messages:
        with st.chat_message(message[LLMStrings.ROLE_ID]):
            st.markdown(message[LLMStrings.CONTENT])

    # React to user input
    if prompt := st.chat_input(LLMStrings.INPUT_PLACEHOLDER):

        # Display user message in chat message container
        with st.chat_message(LLMStrings.USER_ROLE):
            st.markdown(prompt)
            # Add user message to chat history
            user_content = create_message(LLMStrings.USER_ROLE, prompt)
            st.session_state.messages.append(user_content)

        with st.spinner(LLMStrings.WAIT_MESSAGE):
            with st.chat_message(LLMStrings.AI_ROLE):
                # Gửi toàn bộ lịch sử hội thoại cho Gemini từ MongoDB
                response = output_text(llm, prompt, mongo_server)
                
                # Lưu tin nhắn AI vào session state
                ai_content = create_message(LLMStrings.AI_ROLE, response)
                st.session_state.messages.append(ai_content)

                # Hiển thị phản hồi
                simulate_response(response)

                # Lưu vào MongoDB
                mongo_server.insert_many([user_content, ai_content])