from datetime import datetime
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from typing import List, Dict
import os

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
    
    
    def get_recent_messages(self, limit=5):
        '''
        Retrieve the most recent chat history.
        :param limit: Number of recent messages to retrieve.
        :type limit: int
        :return: List of recent messages.
        :rtype: List[Dict]
        '''
        collection = self.get_collection()
    
        # Lấy n tin nhắn gần nhất, sắp xếp theo thời gian giảm dần
        messages = collection.find().sort("_id", -1).limit(limit)
    
        # Chuyển thành list và đảo ngược lại để giữ thứ tự hội thoại
        messages = list(messages)[::-1]
            
        return messages
    
    
    def insert_many(self, data: Dict):
        '''
        Insert multiple data chat to MongoDB.

        :param data: List of Dictionaries.
        :type data: List[Dict]
        '''
        collection = self.get_collection()
        
        try: 
            # Insert the data
            result = collection.insert_many(data)
            print(f"Inserted {len(result.inserted_ids)} documents.")
            print(f"Inserted IDs: {result.inserted_ids}")
        except Exception as e:
            print(f"An error occured: {e}")
        
            
    def init_ttl_index(self):
        """
        Tạo TTL Index cho collection để tự động xóa dữ liệu sau 24h.
        """
        collection = self.get_collection()
        collection.create_index([("timestamp", ASCENDING)], expireAfterSeconds=1000)


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
            {"$set": {"used_tokens": used_tokens, "timestamp": datetime.utcnow()}},
            upsert=True
        )
       
            
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


