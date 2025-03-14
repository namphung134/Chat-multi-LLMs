import streamlit as st
import time
from llm_strings import time_stamp, LLMStrings
import google.generativeai as genai
from openai import OpenAI
from typing import Dict
import os
from db import EasyMongo

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from datetime import datetime
import PyPDF2
from docx import Document
from mistralai import Mistral

# Load API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Cấu hình các model
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
gpt_client = OpenAI(api_key=OPENAI_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Khởi tạo OpenAI client cho embedding
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Mức token tối đa mỗi ngày
MAX_TOKENS_PER_MODEL = 1000

# Function: Check tokens and Get model available
def get_model_and_check_tokens(mongo: EasyMongo, preferred_model=None):
    gemini_usage = mongo.get_token_usage("gemini-2.0-flash")
    gpt_usage = mongo.get_token_usage("gpt-3.5-turbo")
    pixtral_usage = mongo.get_token_usage("pixtral-12b-2409")

    gemini_available = gemini_usage < MAX_TOKENS_PER_MODEL
    gpt_available = gpt_usage < MAX_TOKENS_PER_MODEL
    pixtral_available = pixtral_usage < MAX_TOKENS_PER_MODEL

    if not preferred_model:
        if gemini_available:
            return "gemini-2.0-flash", gemini_model
        elif gpt_available:
            return "gpt-3.5-turbo", gpt_client
        elif pixtral_available:
            return "pixtral-12b-2409", mistral_client
        return None, None

    if preferred_model == "gemini-2.0-flash" and gemini_available:
        return "gemini-2.0-flash", gemini_model
    elif preferred_model == "gpt-3.5-turbo" and gpt_available:
        return "gpt-3.5-turbo", gpt_client
    elif preferred_model == "pixtral-12b-2409" and pixtral_available:
        return "pixtral-12b-2409", mistral_client
    else:
        if gemini_available:
            return "gemini-2.0-flash", gemini_model
        elif gpt_available:
            return "gpt-3.5-turbo", gpt_client
        elif pixtral_available:
            return "pixtral-12b-2409", mistral_client
        return None, None

# Function: Create message with ROLE.ID and CONTENT
def create_message(role: str, content: str) -> Dict:
    return {LLMStrings.ROLE_ID: role, LLMStrings.CONTENT: content}

# FUNCTION: SIMULATE RESPONSE
def simulate_response(text: str):
    message_placeholder = st.empty()
    full_response = ""
    time_delay = 0.05

    for chunk in text.split():
        full_response += chunk + " "
        time.sleep(time_delay)
        message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)


# -----------------------------CRAWL URL-----------------------------------

# FUNCTION: CRAWL DATA FROM URL
def crawl_url(url: str) -> str:
    """Crawl nội dung từ URL, lấy từ nhiều thẻ khác nhau và loại bỏ nội dung không cần thiết."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Loại bỏ các thẻ không chứa nội dung hữu ích
        for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()
        
        # Lấy nội dung từ các thẻ chứa văn bản chính
        content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'div', 'article'])
        text = ' '.join(tag.get_text(strip=True) for tag in content_tags if tag.get_text(strip=True))
        
        if not text:
            return "No meaningful content found on this URL."
        return text.strip()
    except requests.exceptions.RequestException as e:
        return f"Error crawling URL: {str(e)}"
    

# FUNCTION: SPLIT TEXT TO CHUNKS - CHUNKING
def split_text_to_chunks(text: str, chunk_size=500, chunk_overlap=50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


# FUNCTION: EMBEDDING TEXT FROM CHUNKING
def embed_text(text: str) -> list:
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# FUNCTION: URL PROCESSING
def process_url(url: str, mongo: EasyMongo, session_id: str) -> str:
    """Xử lý URL: crawl, split, embed, và lưu vào MongoDB, chỉ nếu chưa tồn tại."""
    collection = mongo.get_database()["VECTORSTORE_COLLECTION"]
    
    existing_doc = collection.find_one({"session_id": session_id, "chunks.url": url, "chunks.type": "url"})
    if existing_doc:
        print(f"URL {url} already processed for session {session_id}. Skipping crawl and embedding.")
        return "URL already processed and stored."
    
    content = crawl_url(url)
    if "Error" in content:
        return content
    
    chunks = split_text_to_chunks(content)
    
    chunk_docs = [
        {
            "chunk_id": i,
            "url": url,
            "type": "url",  # Thêm type để phân biệt
            "content": chunk,
            "embedding": embed_text(chunk),
            "timestamp": datetime.utcnow()
        }
        for i, chunk in enumerate(chunks)
    ]
    
    collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"chunks": {"$each": chunk_docs}},
            "$setOnInsert": {
                "session_id": session_id,
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )
    
    print(f"Processed and stored URL {url} for session {session_id}")
    return "URL processed and stored successfully!"


# FUNCTION: EXTRACT CONTENT (TEXT) FROM FILE (TXT, PDF, DOCX)
def extract_content_from_file(file) -> str:
    """Trích xuất nội dung từ file .txt, .pdf, hoặc .docx."""
    file_name = file.name.lower()
    content = ""
    
    try:
        if file_name.endswith('.txt'):
            content = file.read().decode('utf-8')
        elif file_name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                content += page.extract_text() or ""
        elif file_name.endswith('.docx'):
            doc = Document(file)
            for para in doc.paragraphs:
                content += para.text + "\n"
        else:
            return "Unsupported file format. Please upload .txt, .pdf, or .docx."
        
        if not content.strip():
            return "No content found in the file."
        return content.strip()
    except Exception as e:
        return f"Error processing file: {str(e)}"


# FUNCTION: PROCESSING FILE (TXT, PDF, DOCX)
def process_file(file, mongo: EasyMongo, session_id: str) -> str:
    """Xử lý file: extract, split, embed, và lưu vào MongoDB."""
    collection = mongo.get_database()["VECTORSTORE_COLLECTION"]
    
    file_name = file.name
    # Kiểm tra xem file đã được xử lý trong session_id chưa
    existing_doc = collection.find_one({"session_id": session_id, "chunks.file_name": file_name, "chunks.type": "file_process"})
    if existing_doc:
        print(f"File {file_name} already processed for session {session_id}. Skipping processing.")
        return "File already processed and stored."
    
    content = extract_content_from_file(file)
    if "Error" in content or "No content" in content:
        return content
    
    chunks = split_text_to_chunks(content)
    
    chunk_docs = [
        {
            "chunk_id": i,
            "file_name": file_name,  # Lưu tên file thay vì url
            "type": "file_process",  # Thêm type để phân biệt
            "content": chunk,
            "embedding": embed_text(chunk),
            "timestamp": datetime.utcnow()
        }
        for i, chunk in enumerate(chunks)
    ]
    
    collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"chunks": {"$each": chunk_docs}},
            "$setOnInsert": {
                "session_id": session_id,
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )
    
    print(f"Processed and stored file {file_name} for session {session_id}")
    return "File processed and stored successfully!"


# FUNCTION: CALCULATE COSINE SIMILARITY 
def cosine_similarity(vec1: list, vec2: list) -> float:
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product > 0 else 0


# FUNCTION: RETRIEVING RELEVANT CHUNKS FROM DATABASE
def retrieve_relevant_chunks(session_id: str, identifier: str, query: str, mongo: EasyMongo, type_source: str, top_k=10) -> list:
    """Truy xuất các chunk liên quan từ MongoDB dựa trên embedding của query."""
    query_embedding = embed_text(query)
    collection = mongo.get_database()["VECTORSTORE_COLLECTION"]
    
    doc = collection.find_one({"session_id": session_id})
    if not doc or "chunks" not in doc:
        return []
    
    # Lọc chunks theo type và identifier (url hoặc file_name)
    if type_source == "url":
        chunks = [chunk for chunk in doc["chunks"] if chunk.get("type") == "url" and chunk.get("url") == identifier]
    elif type_source == "file_process":
        chunks = [chunk for chunk in doc["chunks"] if chunk.get("type") == "file_process" and chunk.get("file_name") == identifier]
    else:
        return []
    
    if not chunks:
        return []
    
    scored_chunks = []
    for chunk in chunks:
        similarity = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((similarity, chunk["content"]))
    
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored_chunks[:top_k]]


# Function: OUTPUT TEXT
def output_text(text: str, mongo: EasyMongo, session_id: str, uploaded_file=None, model_name=None) -> str:
    recent_messages = mongo.get_recent_messages(session_id=session_id, limit=5)
    conversation_history = "\n".join(
        [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_messages]
    )

    if uploaded_file:
        process_result = process_file(uploaded_file, mongo, session_id)
        if "Error" in process_result or "No content" in process_result:
            return process_result
        
        file_name = uploaded_file.name
        context_chunks = retrieve_relevant_chunks(session_id, file_name, text, mongo, "file_process")
        context = "\n".join(context_chunks) if context_chunks else "No relevant content found in the file."
        full_prompt = f"""
        You are a smart and friendly AI assistant. Answer questions concisely, clearly, and in a well-structured manner based on the provided context. Do not repeat the user's question.

        Use proper formatting:
        - Use line breaks for readability.
        - If the answer benefits from bullet points, use them appropriately.

        Conversation history:
        {conversation_history}

        Context from file {file_name}:
        {context}

        User query: {text}
        Assistant:
        """
    elif "http" in text.lower():
        url = text.split()[0] if text.split()[0].startswith("http") else None
        query = " ".join(text.split()[1:]) if len(text.split()) > 1 else "Summarize this URL"
        
        if url:
            process_result = process_url(url, mongo, session_id)
            if "Error" in process_result:
                return process_result
            
            context_chunks = retrieve_relevant_chunks(session_id, url, query, mongo, "url")
            context = "\n".join(context_chunks) if context_chunks else "No relevant content found."
            full_prompt = f"""
            You are a smart and friendly AI assistant. Answer questions concisely, clearly, and in a well-structured manner based on the provided context. Do not repeat the user's question.

            Use proper formatting:
            - Use line breaks for readability.
            - If the answer benefits from bullet points, use them appropriately.

            Conversation history:
            {conversation_history}

            Context from URL:
            {context}

            User query: {query}
            Assistant:
            """
        else:
            return "Please provide a valid URL."
    else:
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

    # Sử dụng model_name nếu được truyền vào, nếu không thì lấy từ session_state
    preferred_model = model_name if model_name else st.session_state.get("selected_model", "gemini-2.0-flash")
    actual_model_name, model = get_model_and_check_tokens(mongo, preferred_model=preferred_model)

    if model is None:
        return "Bạn đã dùng hết token hôm nay. Vui lòng quay lại sau 24h!"

    if actual_model_name == "gemini-2.0-flash":
        response = model.generate_content(full_prompt)
        token_used = response._result.usage_metadata.total_token_count
        output_t = response.text
    elif actual_model_name == "gpt-3.5-turbo":
        response = model.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_prompt}])
        token_used = response.usage.total_tokens
        output_t = response.choices[0].message.content
    elif actual_model_name == "pixtral-12b-2409":
        response = mistral_client.chat.complete(
            model=actual_model_name,
            messages=[{"role": "user", "content": full_prompt}]
        )
        token_used = response.usage.total_tokens if hasattr(response, "usage") and hasattr(response.usage, "total_tokens") else 0
        output_t = response.choices[0].message.content
    else:
        return "Model not supported."

    new_usage = mongo.get_token_usage(actual_model_name) + token_used
    mongo.update_token_usage(actual_model_name, new_usage)

    gemini_usage = mongo.get_token_usage("gemini-2.0-flash")
    gpt_usage = mongo.get_token_usage("gpt-3.5-turbo")
    pixtral_usage = mongo.get_token_usage("pixtral-12b-2409")
    gemini_disabled = gemini_usage >= MAX_TOKENS_PER_MODEL
    gpt_disabled = gpt_usage >= MAX_TOKENS_PER_MODEL
    pixtral_disabled = pixtral_usage >= MAX_TOKENS_PER_MODEL

    if actual_model_name == "gemini-2.0-flash" and gemini_disabled:
        if not gpt_disabled:
            st.session_state.selected_model = "gpt-3.5-turbo"
        elif not pixtral_disabled:
            st.session_state.selected_model = "pixtral-12b-2409"
        st.rerun()
    elif actual_model_name == "gpt-3.5-turbo" and gpt_disabled:
        if not gemini_disabled:
            st.session_state.selected_model = "gemini-2.0-flash"
        elif not pixtral_disabled:
            st.session_state.selected_model = "pixtral-12b-2409"
        st.rerun()
    elif actual_model_name == "pixtral-12b-2409" and pixtral_disabled:
        if not gemini_disabled:
            st.session_state.selected_model = "gemini-2.0-flash"
        elif not gpt_disabled:
            st.session_state.selected_model = "gpt-3.5-turbo"
        st.rerun()

    return output_t

# Function: OUTPUT TEXT ALL MODELS
def output_text_all_models(prompt: str, mongo: EasyMongo, session_id: str, uploaded_file=None) -> Dict:
    """Gọi output_text cho tất cả model và trả về kết quả."""
    all_models = ["gemini-2.0-flash", "gpt-3.5-turbo", "pixtral-12b-2409"]
    responses = {}
    
    for model_name in all_models:
        response = output_text(prompt, mongo, session_id, uploaded_file, model_name=model_name)
        responses[model_name] = response
    
    return responses