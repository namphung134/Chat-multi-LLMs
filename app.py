import streamlit as st
from datetime import datetime

from db import EasyMongo
from llm_strings import LLMStrings, time_stamp
from utils import get_model_and_check_tokens, output_text, simulate_response, MAX_TOKENS_PER_MODEL, output_text_all_models

#-------------Kết nối MongoDB (chỉ khởi tạo một lần)----------------
if "mongo_server" not in st.session_state:
    st.session_state.mongo_server = EasyMongo()
    st.session_state.mongo_server.init_ttl_index()  # Khởi tạo TTL Index
    st.session_state.mongo_server.init_vectorstore_ttl_index()  # Khởi tạo TTL Index cho vectorstore

mongo_server = st.session_state.mongo_server

# ------------------------ MODEL SELECTION ------------------------ #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time_stamp())

if "messages" not in st.session_state:
    st.session_state.messages = []

MODEL_OPTIONS = ["gemini-2.0-flash", "gpt-3.5-turbo", "pixtral-12b-2409", "All Models"]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash"  # Mặc định

st.sidebar.title("Model Selection") 

gemini_usage = mongo_server.get_token_usage("gemini-2.0-flash")
gpt_usage = mongo_server.get_token_usage("gpt-3.5-turbo")
pixtral_usage = mongo_server.get_token_usage("pixtral-12b-2409")

gemini_disabled = gemini_usage >= MAX_TOKENS_PER_MODEL
gpt_disabled = gpt_usage >= MAX_TOKENS_PER_MODEL
pixtral_disabled = pixtral_usage >= MAX_TOKENS_PER_MODEL

options_with_status = [
    f"gemini-2.0-flash{' (Hết token)' if gemini_disabled else ''}",
    f"gpt-3.5-turbo{' (Hết token)' if gpt_disabled else ''}",
    f"pixtral-12b-2409{' (Hết token)' if pixtral_disabled else ''}",
    "All Models"
]

if not gemini_disabled:
    default_model = "gemini-2.0-flash"
elif not gpt_disabled:
    default_model = "gpt-3.5-turbo"
elif not pixtral_disabled:
    default_model = "pixtral-12b-2409"
else:
    default_model = None

if "selected_model" not in st.session_state or st.session_state.selected_model not in MODEL_OPTIONS:
    st.session_state.selected_model = default_model

selected_model_with_status = st.sidebar.radio(
    "Chọn Model:",
    options_with_status,
    index=options_with_status.index(
        f"{st.session_state.selected_model}{' (Hết token)' if (st.session_state.selected_model == 'gemini-2.0-flash' and gemini_disabled) or (st.session_state.selected_model == 'gpt-3.5-turbo' and gpt_disabled) or (st.session_state.selected_model == 'pixtral-12b-2409' and pixtral_disabled) else ''}"
    ),
    disabled=(gemini_disabled and gpt_disabled and pixtral_disabled)
)

selected_model = selected_model_with_status.split(" (")[0]

if selected_model == "gemini-2.0-flash" and gemini_disabled:
    if not gpt_disabled:
        st.session_state.selected_model = "gpt-3.5-turbo"
    elif not pixtral_disabled:
        st.session_state.selected_model = "pixtral-12b-2409"
    st.rerun()
elif selected_model == "gpt-3.5-turbo" and gpt_disabled:
    if not gemini_disabled:
        st.session_state.selected_model = "gemini-2.0-flash"
    elif not pixtral_disabled:
        st.session_state.selected_model = "pixtral-12b-2409"
    st.rerun()
elif selected_model == "pixtral-12b-2409" and pixtral_disabled:
    if not gemini_disabled:
        st.session_state.selected_model = "gemini-2.0-flash"
    elif not gpt_disabled:
        st.session_state.selected_model = "gpt-3.5-turbo"
    st.rerun()
elif selected_model != st.session_state.selected_model:
    previous_session_id = st.session_state.session_id
    st.session_state.selected_model = selected_model
    st.session_state.session_id = previous_session_id
    st.rerun()

if gemini_disabled and not gpt_disabled and not pixtral_disabled:
    st.sidebar.warning("⚠️ Model Gemini đã hết token. Đã chuyển sang model khác!")
elif not gemini_disabled and gpt_disabled and not pixtral_disabled:
    st.sidebar.warning("⚠️ Model GPT đã hết token. Đã chuyển sang model khác!")
elif not gemini_disabled and not gpt_disabled and pixtral_disabled:
    st.sidebar.warning("⚠️ Model Pixtral đã hết token. Đã chuyển sang model khác!")
elif gemini_disabled and gpt_disabled and not pixtral_disabled:
    st.sidebar.warning("⚠️ Gemini và GPT hết token. Đã chuyển sang Pixtral!")
elif gemini_disabled and gpt_disabled and pixtral_disabled:
    st.sidebar.warning("⚠️ Tất cả model đều hết token. Vui lòng thử lại sau 24h!")

# -----------------------Giao diện chat------------------------------------------#
st.title(LLMStrings.APP_TITLE)

model_name, model = get_model_and_check_tokens(mongo_server, preferred_model=st.session_state.selected_model)

# if model_name:
#     st.write(f"### Model: {model_name}")
# else:
#     st.write("### ❌ Tất cả model đều hết token!")

# ================================ SIDE BAR ======================================#
st.sidebar.title("Chat Sessions")

chat_sessions = mongo_server.get_chat_sessions() or []

@st.cache_data
def get_cached_messages(session_id):
    return mongo_server.get_recent_messages(session_id) or []

if st.sidebar.button("🆕 New Chat"):
    new_session_id = str(time_stamp())
    st.session_state.session_id = new_session_id
    st.session_state.messages = []

    if new_session_id not in chat_sessions:
        mongo_server.insert_messages(new_session_id, [
            {"session_id": new_session_id, "role": LLMStrings.AI_ROLE, "content": "Hello, how can I help you?", "timestamp": datetime.utcnow()}
        ])
        chat_sessions.append(new_session_id)

    st.rerun()

if chat_sessions:
    selected_session = st.sidebar.radio(
        "History Chat",
        chat_sessions,
        index=chat_sessions.index(st.session_state.session_id) if st.session_state.session_id in chat_sessions else 0
    )

    if selected_session != st.session_state.session_id:
        st.session_state.session_id = selected_session
        st.session_state.messages = get_cached_messages(selected_session)
        st.rerun()

st.sidebar.title("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

#----------------------------------Phần CHATBOT------------------------------------#
if not st.session_state.messages:
    with st.chat_message(LLMStrings.AI_ROLE):
        st.write(LLMStrings.GREETINGS)

# Hiển thị lịch sử tin nhắn
def display_message_group(messages, start_idx):
    """Hiển thị một nhóm tin nhắn (user prompt + AI responses)"""
    user_message = messages[start_idx]
    with st.chat_message(LLMStrings.USER_ROLE):
        st.markdown(user_message["content"])
    
    # Thu thập các phản hồi AI liên quan
    ai_responses = []
    i = start_idx + 1
    # Kiểm tra các tin nhắn AI ngay sau tin nhắn user, không cần so sánh timestamp chính xác
    while i < len(messages) and messages[i]["role"] == LLMStrings.AI_ROLE:
        ai_responses.append(messages[i])
        i += 1
    
    if len(ai_responses) > 1:  # Nếu có nhiều phản hồi (từ All Models)
        st.markdown("""
            <style>
            .chat-container {
                width: 100%;
                overflow-x: auto;
                white-space: nowrap;
            }
            .chat-column {
                display: inline-block;
                width: 33%;
                vertical-align: top;
                padding: 10px;
                white-space: normal;
            }
            </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            cols = st.columns(3)
            for idx, response in enumerate(ai_responses[:3]):  # Giới hạn 3 cột
                with cols[idx]:
                    st.markdown(f"**{response['model_name']}**:")
                    st.markdown(response["content"])
    else:  # Nếu chỉ có một phản hồi
        with st.chat_message(LLMStrings.AI_ROLE):
            if ai_responses and "model_name" in ai_responses[0]:
                st.markdown(f"**{ai_responses[0]['model_name']}**: {ai_responses[0]['content']}")
            elif ai_responses:
                st.markdown(ai_responses[0]["content"])

# Hiển thị lịch sử tin nhắn
i = 0
while i < len(st.session_state.messages):
    if st.session_state.messages[i]["role"] == LLMStrings.USER_ROLE:
        display_message_group(st.session_state.messages, i)
        i += 1
        while i < len(st.session_state.messages) and st.session_state.messages[i]["role"] == LLMStrings.AI_ROLE:
            i += 1
    else:
        i += 1

# Xử lý tin nhắn mới
if prompt := st.chat_input(LLMStrings.INPUT_PLACEHOLDER):
    with st.chat_message(LLMStrings.USER_ROLE):
        st.markdown(prompt)
        user_timestamp = time_stamp()  # Timestamp chung cho cả nhóm
        st.session_state.messages.append({"role": LLMStrings.USER_ROLE, "content": prompt, "timestamp": user_timestamp})

    with st.spinner(LLMStrings.WAIT_MESSAGE):
        if st.session_state.selected_model == "All Models":
            responses = output_text_all_models(prompt, mongo_server, st.session_state.session_id, uploaded_file)
            
            for model_name, response in responses.items():
                st.session_state.messages.append({
                    "role": LLMStrings.AI_ROLE, 
                    "content": response, 
                    "model_name": model_name,
                    "timestamp": user_timestamp  # Dùng cùng timestamp với user
                })
            
            st.markdown("""
                <style>
                .chat-container { width: 100%; overflow-x: auto; white-space: nowrap; }
                .chat-column { display: inline-block; width: 33%; vertical-align: top; padding: 10px; white-space: normal; }
                </style>
            """, unsafe_allow_html=True)
            with st.container():
                cols = st.columns(3)
                for i, (model_name, response) in enumerate(responses.items()):
                    with cols[i]:
                        st.markdown(f"**{model_name}**:")
                        simulate_response(response)
            
            mongo_server.insert_messages(st.session_state.session_id, [
                {"role": LLMStrings.USER_ROLE, "content": prompt, "timestamp": user_timestamp},
                *[
                    {"role": LLMStrings.AI_ROLE, "content": response, "model_name": model_name, "timestamp": user_timestamp}
                    for model_name, response in responses.items()
                ]
            ])
        else:
            with st.chat_message(LLMStrings.AI_ROLE):
                response = output_text(prompt, mongo_server, st.session_state.session_id, uploaded_file)
                st.session_state.messages.append({"role": LLMStrings.AI_ROLE, "content": response, "timestamp": user_timestamp})
                simulate_response(response)

            mongo_server.insert_messages(st.session_state.session_id, [
                {"role": LLMStrings.USER_ROLE, "content": prompt, "timestamp": user_timestamp},
                {"role": LLMStrings.AI_ROLE, "content": response, "timestamp": user_timestamp}
            ])