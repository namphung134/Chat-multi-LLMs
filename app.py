import streamlit as st
from datetime import datetime

from db import EasyMongo
from llm_strings import LLMStrings, time_stamp
from utils import get_model_and_check_tokens, output_text, simulate_response, MAX_TOKENS_PER_MODEL

#-------------Kết nối MongoDB (chỉ khởi tạo một lần)----------------
if "mongo_server" not in st.session_state:
    st.session_state.mongo_server = EasyMongo()
    st.session_state.mongo_server.init_ttl_index()  # Khởi tạo TTL Index
    st.session_state.mongo_server.init_vectorstore_ttl_index()  # Khởi tạo TTL Index cho vectorstore

mongo_server = st.session_state.mongo_server


# ------------------------ MODEL SELECTION ------------------------ #

# Nếu chưa có session_id trong session_state, khởi tạo mặc định
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time_stamp())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Danh sách model hỗ trợ
MODEL_OPTIONS = ["gemini-2.0-flash", "gpt-3.5-turbo", "pixtral-12b-2409"]

# Lấy model đang sử dụng
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash"  # Mặc định

st.sidebar.title("Model Selection") 

# Kiểm tra token usage của từng model
gemini_usage = mongo_server.get_token_usage("gemini-2.0-flash")
gpt_usage = mongo_server.get_token_usage("gpt-3.5-turbo")
pixtral_usage = mongo_server.get_token_usage("pixtral-12b-2409")

# Xác định trạng thái của từng model
gemini_disabled = gemini_usage >= MAX_TOKENS_PER_MODEL
gpt_disabled = gpt_usage >= MAX_TOKENS_PER_MODEL
pixtral_disabled = pixtral_usage >= MAX_TOKENS_PER_MODEL

# Tạo danh sách options với nhãn tùy chỉnh để hiển thị trạng thái
options_with_status = [
    f"gemini-2.0-flash{' (Hết token)' if gemini_disabled else ''}",
    f"gpt-3.5-turbo{' (Hết token)' if gpt_disabled else ''}",
    f"pixtral-12b-2409{' (Hết token)' if pixtral_disabled else ''}"
]

# Xác định model mặc định dựa trên token còn lại
# if gemini_disabled and not gpt_disabled:
#     default_model = "gpt-3.5-turbo"
# elif gpt_disabled and not gemini_disabled:
#     default_model = "gemini-2.0-flash"
# else:
#     default_model = "gemini-2.0-flash" if not gemini_disabled else "gpt-3.5-turbo"

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

# Radio chọn model trên sidebar
selected_model_with_status = st.sidebar.radio(
    "Chọn Model:",
    options_with_status,
    index=options_with_status.index(
        f"{st.session_state.selected_model}{' (Hết token)' if (st.session_state.selected_model == 'gemini-2.0-flash' and gemini_disabled) or (st.session_state.selected_model == 'gpt-3.5-turbo' and gpt_disabled) or (st.session_state.selected_model == 'pixtral-12b-2409' and pixtral_disabled) else ''}"
    ),
    disabled=(gemini_disabled and gpt_disabled and pixtral_disabled)
)

# Lấy model thực tế từ lựa chọn (loại bỏ phần trạng thái)
selected_model = selected_model_with_status.split(" (")[0]

# Tự động chuyển nếu model được chọn hết token
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

# Warning khi LLM Models hết token
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
#--------------------------------------------------------------------------------# 


# -----------------------Giao diện chat------------------------------------------#

# Giao diện ứng dụng
st.title(LLMStrings.APP_TITLE)

# Lấy model đang sử dụng
model_name, model = get_model_and_check_tokens(mongo_server, preferred_model=st.session_state.selected_model)

if model_name:
    st.write(f"### Model: {model_name}")
else:
    st.write("### ❌ Tất cả model đều hết token!")

#---------------------------------------------------------------------------------#


# ================================ SIDE BAR ======================================#
st.sidebar.title("Chat Sessions")

# Lấy danh sách session từ MongoDB
chat_sessions = mongo_server.get_chat_sessions() or []

# ------------------------------ Caching dữ liệu -------------------------------- #
@st.cache_data
def get_cached_messages(session_id):
    return mongo_server.get_recent_messages(session_id) or []

# ------------------------ New Chat - Tạo session mới --------------------------- #

if st.sidebar.button("🆕 New Chat"):
    new_session_id = str(time_stamp())  # Tạo session_id mới
    st.session_state.session_id = new_session_id
    st.session_state.messages = []  # Reset tin nhắn tránh bị duplicate

    # Chỉ thêm session vào DB nếu chưa tồn tại
    if new_session_id not in chat_sessions:
        mongo_server.insert_messages(new_session_id, [
            {"session_id": new_session_id, "role": LLMStrings.AI_ROLE, "content": "Hello, how can I help you?", "timestamp": datetime.utcnow()}
        ])
        chat_sessions.append(new_session_id)

    # Cập nhật UI ngay lập tức
    st.rerun()

# ---------------------------- Chọn session từ lịch sử ---------------------------- #

if chat_sessions:
    selected_session = st.sidebar.radio(
        "History Chat",
        chat_sessions,
        index=chat_sessions.index(st.session_state.session_id) if st.session_state.session_id in chat_sessions else 0
    )

    if selected_session != st.session_state.session_id:
        st.session_state.session_id = selected_session
        st.session_state.messages = get_cached_messages(selected_session)  # Dùng caching để giảm query
        st.rerun()

#-------------------------Thêm phần upload file vào sidebar------------------------ #

st.sidebar.title("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

#================================================================================== #



#----------------------------------Phần CHATBOT------------------------------------ #
# Hiển thị lời chào từ AI nếu chưa có tin nhắn
if not st.session_state.messages:
    with st.chat_message(LLMStrings.AI_ROLE):
        st.write(LLMStrings.GREETINGS)

# Hiển thị lịch sử tin nhắn trong session
for message in st.session_state.messages:
    role = message["role"]
    if role not in [LLMStrings.USER_ROLE, LLMStrings.AI_ROLE]:
        role = LLMStrings.AI_ROLE
    with st.chat_message(role):
        st.markdown(message["content"])

# Xử lý tin nhắn đầu vào từ người dùng
if prompt := st.chat_input(LLMStrings.INPUT_PLACEHOLDER):
    with st.chat_message(LLMStrings.USER_ROLE):
        st.markdown(prompt)
        st.session_state.messages.append({"role": LLMStrings.USER_ROLE, "content": prompt})

    with st.spinner(LLMStrings.WAIT_MESSAGE):
        with st.chat_message(LLMStrings.AI_ROLE):
            # Truyền uploaded_file vào output_text nếu có
            response = output_text(prompt, mongo_server, session_id=st.session_state.session_id, uploaded_file=uploaded_file)
            st.session_state.messages.append({"role": LLMStrings.AI_ROLE, "content": response})
            simulate_response(response)

        mongo_server.insert_messages(st.session_state.session_id, [
            {"role": LLMStrings.USER_ROLE, "content": prompt, "timestamp": time_stamp()},
            {"role": LLMStrings.AI_ROLE, "content": response, "timestamp": time_stamp()}
        ])

#---------------------------------------------------------------------------------#
