import google.generativeai as genai
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

from db import EasyMongo
from llm_strings import LLMStrings
from utils import output_text, simulate_response, get_model_and_check_tokens

# Load API key từ môi trường
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Kết nối MongoDB
mongo_server = EasyMongo()
mongo_server.init_ttl_index()  # Khởi tạo TTL Index

# Giao diện ứng dụng
st.title(LLMStrings.APP_TITLE)

# Lấy tên model để hiển thị trong chat
model_name, model = get_model_and_check_tokens(mongo_server)
st.write(f"### Model: {model_name}")

# Hiển thị lời chào từ AI
with st.chat_message(LLMStrings.AI_ROLE):
    st.write(LLMStrings.GREETINGS)

# Khởi tạo session state nếu chưa có
if LLMStrings.SESSION_STATES not in st.session_state:
    st.session_state.messages = []

# Lấy lịch sử hội thoại từ MongoDB (chỉ lấy một lần)
if not st.session_state.messages:
    messages = mongo_server.get_collection().find()
    st.session_state.messages = [
        {"role": message[LLMStrings.ROLE_ID], "content": message[LLMStrings.CONTENT]}
        for message in messages
    ]

# Hiển thị tin nhắn trước đó
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý đầu vào từ user
if prompt := st.chat_input(LLMStrings.INPUT_PLACEHOLDER):
    # Hiển thị tin nhắn của user
    with st.chat_message(LLMStrings.USER_ROLE):
        st.markdown(prompt)
        st.session_state.messages.append({"role": LLMStrings.USER_ROLE, "content": prompt})

    with st.spinner(LLMStrings.WAIT_MESSAGE):
        with st.chat_message(LLMStrings.AI_ROLE):
            response = output_text(prompt, mongo_server)

            # Lưu phản hồi AI vào session state
            st.session_state.messages.append({"role": LLMStrings.AI_ROLE, "content": response})

            # Hiển thị phản hồi
            simulate_response(response)


        # Lưu vào MongoDB với timestamp
        mongo_server.insert_many([
            {
                "role": LLMStrings.USER_ROLE,
                "content": prompt,
                "timestamp": datetime.utcnow()  # Thêm timestamp
            },
            {
                "role": LLMStrings.AI_ROLE,
                "content": response,
                "timestamp": datetime.utcnow()
            }
        ])
