import streamlit as st
from datetime import datetime

from db import EasyMongo
from llm_strings import LLMStrings, time_stamp
from utils import get_model_and_check_tokens, output_text, simulate_response

# Kết nối MongoDB
mongo_server = EasyMongo()
mongo_server.init_ttl_index()  # Khởi tạo TTL Index

# Giao diện ứng dụng
st.title(LLMStrings.APP_TITLE)

# Lấy tên model để hiển thị trong chat
model_name, model = get_model_and_check_tokens(mongo_server)
st.write(f"### Model: {model_name}")

# ------------------------ SIDE BAR ------------------------ #
st.sidebar.title("Chat Sessions")

# Lấy danh sách session từ MongoDB
chat_sessions = mongo_server.get_chat_sessions() or []

# Nếu chưa có session_id trong session_state, khởi tạo mặc định
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time_stamp())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------ Tạo session mới ------------------------ #
if st.sidebar.button("🆕 New Chat"):
    new_session_id = str(time_stamp())  # Tạo session_id mới
    # new_session_id = datetime.utcnow().isoformat()
    st.session_state.session_id = new_session_id
    st.session_state.messages = []  # Reset tin nhắn tránh bị duplicate

    # Thêm session vào DB nếu chưa có
    if new_session_id not in chat_sessions:
        mongo_server.insert_messages(new_session_id, [
            {"session_id": new_session_id, "role": LLMStrings.AI_ROLE, "content": "Hello, how can I help you?", "timestamp": datetime.utcnow()}
        ])

    # Cập nhật danh sách session
    chat_sessions.append(new_session_id)

    # Cập nhật UI ngay lập tức
    st.rerun()

# ------------------------ Chọn session từ lịch sử ------------------------ #
if chat_sessions:
    selected_session = st.sidebar.radio(
        "History Chat",
        chat_sessions,
        index=chat_sessions.index(st.session_state.session_id) if st.session_state.session_id in chat_sessions else 0
    )

    # Chỉ load lại khi chọn session mới (tránh load trùng)
    if selected_session != st.session_state.session_id:
        st.session_state.session_id = selected_session

        # **Xóa tin nhắn cũ trước khi tải tin nhắn mới**
        st.session_state.messages = []

        messages = mongo_server.get_recent_messages(selected_session)
        if messages:
            st.session_state.messages = messages

        # Rerun UI để cập nhật giao diện
        st.rerun()

#---------------------------------------------------------#

# Hiển thị lời chào từ AI nếu chưa có tin nhắn nào
if not st.session_state.messages:
    with st.chat_message(LLMStrings.AI_ROLE):
        st.write(LLMStrings.GREETINGS)

# Hiển thị lịch sử tin nhắn trong session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý tin nhắn đầu vào từ người dùng
if prompt := st.chat_input(LLMStrings.INPUT_PLACEHOLDER):
    with st.chat_message(LLMStrings.USER_ROLE):
        st.markdown(prompt)
        st.session_state.messages.append({"role": LLMStrings.USER_ROLE, "content": prompt})

    with st.spinner(LLMStrings.WAIT_MESSAGE):
        with st.chat_message(LLMStrings.AI_ROLE):
            response = output_text(prompt, mongo_server, session_id=st.session_state.session_id)
            st.session_state.messages.append({"role": LLMStrings.AI_ROLE, "content": response})
            simulate_response(response)
        
        mongo_server.insert_messages(st.session_state.session_id, [
            {"role": LLMStrings.USER_ROLE, "content": prompt, "timestamp": time_stamp()},
            {"role": LLMStrings.AI_ROLE, "content": response, "timestamp": time_stamp()}
        ])