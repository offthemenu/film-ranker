import os
import requests
import pymongo
from pymongo.errors import ServerSelectionTimeoutError
import certifi
from tqdm import tqdm
from dotenv import load_dotenv

# MongoDB Configuration
load_dotenv() 
mongodb_uri = os.getenv('Mongo_URI') #retrieve mongodb uri from .env file

# load desired directory for storing poster images
image_directory = "./training-models/posters/fromDB"
os.makedirs(image_directory, exist_ok=True)

try:
    db_client = pymongo.MongoClient(mongodb_uri, tlsCAFile=certifi.where()) # this creates a client that can connect to our DB
    db = db_client.get_database("movies") # this gets the database named 'Movies'
    movieDetails = db.get_collection("movieDetails")

    db_client.server_info() # forces client to connect to server
    print("Connected successfully to the 'Movies' database!")
    
    posterDetails = db.get_collection("posterDetails")
    print("Connected successfully to the 'Posters' database!")

except pymongo.errors.ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    exit(1)


