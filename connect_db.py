from pymongo import MongoClient

# Kết nối đến MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Truy cập vào cơ sở dữ liệu
db = client['MONGO_DB']

# Truy cập vào collection
collection = db['MONGO_COLLECTION']

# Thực hiện các thao tác với collection
# Ví dụ: Lấy tất cả các document trong collection
documents = collection.find()
for doc in documents:
    print(doc)