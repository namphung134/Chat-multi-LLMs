import streamlit as st
from db import save_chat, load_chat, get_all_sessions
from chatbot import get_gemini_response
from utils import display_chat

# Setup giao diện
st.set_page_config(page_title="Chatbot Gemini", layout="wide")
st.sidebar.title("Chatbot Gemini")

# Quản lý session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# Load danh sách lịch sử chat
st.sidebar.subheader("History")
session_list = get_all_sessions()
selected_session = st.sidebar.selectbox("Chọn cuộc trò chuyện", session_list, index=0) if session_list else None

if st.sidebar.button("New Chat"):
    st.session_state.session_id = f"chat_{len(session_list) + 1}"
    st.session_state.messages = []

# Load lịch sử chat từ MongoDB
if selected_session:
    st.session_state.session_id = selected_session
    st.session_state.messages = load_chat(selected_session)

# Hiển thị phần chat
st.title("🤖 Chatbot Gemini")
st.subheader("Trò chuyện cùng AI")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

display_chat(st.session_state.messages)

# Nhận input từ người dùng
user_input = st.text_input("Nhập tin nhắn...", key="user_input")

if st.button("Gửi"):
    if user_input:
        # Thêm tin nhắn người dùng vào session
        st.session_state.messages.append({"role": "user", "text": user_input})

        # Gọi Gemini API
        bot_response = get_gemini_response(user_input)
        st.session_state.messages.append({"role": "bot", "text": bot_response})

        # Lưu vào MongoDB
        save_chat(st.session_state.session_id, st.session_state.messages)

        # Cập nhật giao diện chat
        st.experimental_rerun()
