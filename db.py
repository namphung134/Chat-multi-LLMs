from datetime import datetime
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from typing import List, Dict
import os
from llm_strings import LLMStrings, time_stamp


# take environment variables from .env
load_dotenv()


class EasyMongo:
    '''
    A simple wrapper for MongoDB Clients.
    '''
    

    def __init__(self):
        '''
        Initialize the MongoDB client.
        '''
        self.URI = os.getenv("MONGO_URI")
        self.DB = "MONGO_DB"
        self.COLLECTION = "MONGO_COLLECTION"
        self.VECTORSTORE_COLLECTION = "VECTORSTORE_COLLECTION"  # Collection mới cho vectorstore
        
    def get_database(self):
        '''
        Create a connection to the MongoDB Atlas url and return NoSQL Database.
        '''
        client = MongoClient(self.URI)
        
        # Connect to the database
        db = client[self.DB]
        return db
    
    
    def get_collection(self):
        '''
        Get the collection from the database.
        '''
        dbname = self.get_database()
        
        collection = dbname[self.COLLECTION]
        return collection


    def get_recent_messages(self, session_id: str, limit=10):
        """
        Retrieve the most recent messages for a given session_id.
        """
        collection = self.get_collection()
        session = collection.find_one({"session_id": session_id}, {"messages": {"$slice": -limit}})
        
        if not session:
            print(f"Warning: No messages found for session_id {session_id}")
            return []
        
        return session["messages"]


    def insert_messages(self, session_id: str, messages: List[Dict]):
        """
        Chèn hoặc cập nhật tin nhắn vào session_id.
        Nếu session chưa tồn tại, tạo mới với ObjectId().
        """
        if not session_id:
            raise ValueError("session_id không hợp lệ!")

        collection = self.get_collection()

        try:
            collection.update_one(
                {"session_id": session_id},  # Truy vấn theo session_id
                {
                    "$push": {"messages": {"$each": messages}}, 
                    "$setOnInsert": {
                        "session_id": session_id,  # Đảm bảo session_id luôn tồn tại
                        "created_at": time_stamp()
                    }
                },
                upsert=True
            )
            print(f"Updated session {session_id} with {len(messages)} new messages")
            print(collection)
        except Exception as e:
            print(f"An error occurred: {e}")


    def get_chat_sessions(self):
        """
        Retrieve all unique chat session IDs.
        """
        collection = self.get_collection()
        sessions = collection.distinct("session_id")  # Lấy tất cả session_id duy nhất
        return sessions


    def get_token_usage(self, model_name):
        """
        Lấy số token đã sử dụng của mô hình từ MongoDB.
        """
        collection = self.get_collection()
        usage = collection.find_one({"type": "token_usage", "model": model_name})
        return usage["used_tokens"] if usage else 0


    def update_token_usage(self, model_name, used_tokens):
        """
        Cập nhật số token đã sử dụng vào MONGO_COLLECTION.
        """
        collection = self.get_collection()
        collection.update_one(
            {"type": "token_usage", "model": model_name},
            {"$set": {
                "used_tokens": used_tokens, 
                "timestamp": datetime.utcnow() #không dùng time_stamp() vì nó sẽ không xoá đc token_usage
                }
            },
            upsert=True
        )
       

    #----------------------URL---------------------------

    def init_ttl_index(self):
        """Tạo TTL Index cho token usage trong MONGO_COLLECTION."""
        collection = self.get_collection()
        existing_indexes = collection.index_information()
        for index in existing_indexes:
            if "timestamp" in existing_indexes[index]["key"]:
                collection.drop_index(index)
        collection.create_index(
            [("timestamp", ASCENDING)],
            expireAfterSeconds=300,  # 1 ngày
            partialFilterExpression={"type": "token_usage"}
        )


    def init_vectorstore_ttl_index(self):
        collection = self.get_database()[self.VECTORSTORE_COLLECTION]
        index_name = "created_at_1"
        
        existing_indexes = collection.index_information()
        if index_name in existing_indexes:
            if existing_indexes[index_name].get("expireAfterSeconds") != 300:  # 300 giây như code hiện tại của bạn
                collection.drop_index(index_name)
                print(f"Dropped existing index {index_name} in {self.VECTORSTORE_COLLECTION}")
        
        collection.create_index(
            [("created_at", ASCENDING)],
            expireAfterSeconds=300,
            name=index_name
        )
        print(f"Created TTL index for {self.VECTORSTORE_COLLECTION} on created_at")


    def test_data(self):
        '''
        Test data for MongoDB.
        '''
        user_content = {"role": "user", "content": "What is machine learning in 200 characters?"}
        ai_content = {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that "
                                                      "enables computers to learn and improve their performance on a "
                                                      "task without explicitly programmed instructions, by using "
                                                      "algorithms and statistical models to analyze and learn "
                                                      "from data."}
        user_content2 = {"role": "user", "content": "What is deep learning in 200 characters?"}
        ai_content2 = {"role": "assistant", "content": "Deep learning is a subset of machine learning that utilizes "
                                                       "neural networks with multiple layers to learn and represent "
                                                       "complex patterns in data. It enables AI models to recognize "
                                                       "and make decisions based on intricate relationships within "
                                                       "the data, leading to improved accuracy and efficiency in "
                                                       "various applications such as image recognition, natural "
                                                       "language processing, and speech recognition."}
        
        self.insert_many([user_content, ai_content, user_content2, ai_content2])


