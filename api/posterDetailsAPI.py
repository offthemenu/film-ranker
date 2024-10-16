from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic import Field
from typing import List, Optional, Union, Dict
import pymongo
import certifi
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os
from typing import Union, List

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Fetch MongoDB URI from environment variables
try:
    mongo_uri = os.getenv("Mongo_URI")
    client = pymongo.MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client["movies"]
    movieDetails = db["movieDetails"]
    posterDetails = db["posterDetails"]

    client.server_info() # forces client to connect to server
    print("Connected successfully to the 'Movies' database!")

except pymongo.errors.ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    exit(1)

# Root endpoint to welcome users to the API
@app.get("/")
def read_root():
    return {"message": "Welcome to the Movies API. Visit /movies to explore the collection."}

# Pydantic model for poster characteristics
class PosterCharacteristics(BaseModel):
    imdbID: str
    posterLink: str
    title: Optional[str]
    tagline: Optional[str]
    colorScheme: Optional[List[str]]
    font: Optional[List[str]]
    atmosphere: Optional[str]
    imageElement: Optional[Dict[str, str]]
    artStyle: Optional[str]
    periodStyle: Optional[str]

# Add poster characteristics by IMDb ID
@app.post("/posters/{imdbID}", response_model=PosterCharacteristics)
def add_poster_characteristics(imdbID: str, poster_characteristics: PosterCharacteristics):
    imdbID = imdbID.strip()

    # Query movieDetails collection to ensure the movie exists and has a valid posterLink
    movie = movieDetails.find_one({"imdbID": imdbID})

    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found in movieDetails collection.")
    
    if movie.get('posterLink') == "N/A":
        raise HTTPException(status_code=400, detail="Movie has no valid poster link (posterLink is 'N/A').")
    
    # Insert poster characteristics into the posterDetails collection
    poster_data = poster_characteristics.model_dump(exclude_unset=True)

    # Check if poster details already exist for this movie
    existing_poster = posterDetails.find_one({"imdbID": imdbID})
    if existing_poster:
        raise HTTPException(status_code=400, detail="Poster details already exist for this IMDb ID.")
    
    posterDetails.insert_one(poster_data)

    return poster_characteristics