from pymongo import MongoClient
from config import MONGO_URI

DB_NAME = "chatbot_gemini_db"
COLLECTION_NAME = "chats"

# Kết nối đến MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def save_chat(session_id, messages):
    """Lưu lịch sử chat vào MongoDB"""
    collection.update_one({"session_id": session_id}, {"$set": {"messages": messages}}, upsert=True)

def load_chat(session_id):
    """Tải lịch sử chat từ MongoDB"""
    chat = collection.find_one({"session_id": session_id})
    return chat["messages"] if chat else []
    
def get_all_sessions():
    """Lấy danh sách các phiên chat"""
    return [chat["session_id"] for chat in collection.find({}, {"session_id": 1})]
