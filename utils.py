import streamlit as st
import time
from llm_strings import LLMStrings
import google.generativeai as genai
from openai import OpenAI
from typing import Dict
import os
from db import EasyMongo


# Load API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cấu hình các model
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
gpt_client = OpenAI(api_key=OPENAI_API_KEY)

# Mức token tối đa mỗi ngày
MAX_TOKENS_PER_MODEL = 2000


def get_model_and_check_tokens(mongo: EasyMongo):
    """
    Kiểm tra model nào còn token để sử dụng.
    """
    gemini_usage = mongo.get_token_usage("gemini-2.0-flash")
    gpt_usage = mongo.get_token_usage("gpt-3.5-turbo")

    if gemini_usage < MAX_TOKENS_PER_MODEL:
        return "gemini-2.0-flash", gemini_model
    elif gpt_usage < MAX_TOKENS_PER_MODEL:
        return "gpt-3.5-turbo", gpt_client
    else:
        return None, None


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


def output_text(text: str, mongo: EasyMongo) -> str:
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
    print(recent_messages)
    
    # Xây dựng lịch sử hội thoại
    conversation_history = "\n".join(
        [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_messages]
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
    
    model_name, model = get_model_and_check_tokens(mongo)

    if model is None:
        return "Bạn đã dùng hết token hôm nay. Vui lòng quay lại sau 24h!"

    # Tạo phản hồi từ mô hình tương ứng
    if model_name == "gemini-2.0-flash":
        response = model.generate_content(full_prompt)
        print(response)
        token_used = response._result.usage_metadata.total_token_count
        output_t = response.text
        
    else:  # GPT-3.5-turbo
        response = model.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_prompt}])
        token_used = response.usage.total_tokens
        output_t = response.choices[0].message.content

    # Cập nhật token usage vào MongoDB
    new_usage = mongo.get_token_usage(model_name) + token_used
    mongo.update_token_usage(model_name, new_usage)

    return output_t  # Trả về kết quả sau khi đã cập nhật token



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