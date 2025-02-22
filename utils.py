import streamlit as st
import time
from llm_strings import LLMStrings
import google.generativeai as genai
from typing import Dict
import os
from db import EasyMongo


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Đặt API key trên Streamlit Cloud

gemini_model = genai.GenerativeModel("gemini-2.0-flash")


def create_message(role: str, content: str) -> Dict:
    """
    :param role: Role of the message sender, i.e. ai, user, assistant.
    :type role: str
    
    :param content: Content of the message.
    :type content: str

    :return: Message data.
    :rtype: Dict
    """
    return {LLMStrings.ROLE_ID: role, LLMStrings.CONTENT: content}


def output_text(llm_model: genai.GenerativeModel, text: str, mongo: EasyMongo) -> str:
    """
    Generates output from the LLM model.

    :param llm_model: LLM Model.
    :type llm_model: genai.GenerativeModel
    :param text: Input text prompt.
    :type text: str
    :param mongo: EasyMongo instance to fetch chat history.
    :type mongo: EasyMongo

    :return: LLM output - the generated text.
    :rtype: str
    """
    # Lấy 5 tin nhắn gần nhất từ MongoDB
    recent_messages = mongo.get_recent_messages(limit=5)

    # Xây dựng lịch sử hội thoại
    conversation_history = "\n".join(
        [f"{msg[LLMStrings.ROLE_ID]}: {msg[LLMStrings.CONTENT]}" for msg in recent_messages]
    )

    # Ghép prompt với lịch sử hội thoại
    full_prompt = f"""
    You are a smart and friendly AI assistant. Answer questions concisely, clearly, and in a well-structured manner. Do not repeat the user's question.

    Use proper formatting:
    - Use line breaks for readability.
    - If the answer benefits from bullet points, use them appropriately.

    Conversation history:
    {conversation_history}

    User: {text}
    Assistant:
    """
    # print(f"Full prompt: {full_prompt}")
    
    response = llm_model.generate_content(full_prompt)

    return response.text


def simulate_response(text: str):
    """
    Simulate stream of response with milliseconds delay.

    :param text: LLM response text.
    :type text: str
    """
    message_placeholder = st.empty()
    full_response = ""
    time_delay = 0.05

    for chunk in text.split():
        full_response += chunk + " "
        time.sleep(time_delay)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")

    # Write full response
    message_placeholder.markdown(full_response)