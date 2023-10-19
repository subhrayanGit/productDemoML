import pymongo
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from urllib.parse import quote_plus

# Connect to MongoDB
#client = pymongo.MongoClient("mongodb://"mongoAdmin":"mongoDCP@1"@35.178.144.244:27017/productDB")

# Connection settings
host = "35.178.144.244"  # Replace with your MongoDB server address
port = 27017  # Default MongoDB port
username = "mongoAdmin"  # Replace with your MongoDB username
password = "mongoDCP@1"  # Replace with your MongoDB password
auth_source = 'admin'
database_name = "productDB"  # Replace with your database name




# Escape the username and password using quote_plus
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

# Create a MongoDB URI with escaped username and password
mongo_uri = uri = f"mongodb://{escaped_username}:{escaped_password}@{host}:{port}/{auth_source}"

# Create a MongoDB client
client = pymongo.MongoClient(mongo_uri)

# Access the database and collection as needed
db = client.productDB

# Access the collection and retrieve data
collection = db.salesData
documents = list(collection.find())

# Print the documents
print(len(documents))

# Close the connection
client.close()


