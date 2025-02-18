import streamlit as st


def main():
    st.set_page_config(page_title="Multi-LLM Chatbot", page_icon=":shark:", layout="wide")
    
    with st.sidebar:
        st.title("Multi-LLM Chatbot")
        st.write("Welcome to the Multi-LLM Chatbot! This chatbot is powered by multiple language models, including GPT, GEMINI, and more.")

    
    #main content area for display chat message
    st.title("Chat with multiple LLMs")
    st.write("This is the main content area for the chat messages")
    

    #chat input
    if "message" not in st.session.keys():
        st.session.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if 
