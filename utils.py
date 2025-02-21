import streamlit as st

def format_message(role, text):
    """Định dạng tin nhắn hiển thị"""
    if role == "user":
        return f"👤 **You:** {text}"
    return f"🤖 **Chatbot:** {text}"

def display_chat(messages):
    """Hiển thị lịch sử chat"""
    for msg in messages:
        st.markdown(format_message(msg["role"], msg["text"]))
