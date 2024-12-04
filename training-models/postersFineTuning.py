import os
import pymongo
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import certifi
from tqdm import tqdm

# Load environment variables
load_dotenv()
mongodb_uri = os.getenv('Mongo_URI')

# Connect to MongoDB
client = pymongo.MongoClient(mongodb_uri, tlsCAFile=certifi.where())
db = client["movies"]
movieDetails = db.get_collection("movieDetails")

# Initialize Stable Diffusion pipeline and tokenizer
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define the PosterDataset class
class PosterDataset(Dataset):
    def __init__(self, documents):
        """
        Initialize with a list of documents from MongoDB. Each document contains
        'trainingPrompt' and 'posterLink' fields for the prompt and image URL.
        """
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        """
        Fetches and processes an item by its index, including:
        - Downloading the poster image
        - Tokenizing the prompt
        """
        doc = self.documents[idx]
        image_url = doc.get("posterLink")
        prompt = doc.get("trainingPrompt")

        if not image_url or not prompt:
            raise ValueError(f"Missing data for document ID: {doc['_id']}")

        # Fetch image from URL
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)

        # Process image with the pipeline's feature extractor
        pixel_values = pipeline.feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"].squeeze(0)
        }

# Function to load a batch of documents
def load_batch(max_batch_size):
    """
    Load a batch of documents for training.
    """
    pipeline = [
        {"$match": {"trainingPrompt": {"$exists": True}, "posterLink": {"$exists": True}}},
        {"$sample": {"size": max_batch_size}}
    ]
    documents = list(movieDetails.aggregate(pipeline))
    print(f"Loaded {len(documents)} documents for training.")
    return documents if documents else None

# Training loop
def train_on_batch(documents, optimizer, loss_fn):
    """
    Trains the model on a single batch of documents using PyTorch.
    """
    # Initialize dataset and dataloader for the batch
    dataset = PosterDataset(documents)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    pipeline.unet.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass
        pixel_values = batch["pixel_values"].to("cuda")
        input_ids = batch["input_ids"].to("cuda")

        noise = torch.randn_like(pixel_values).to("cuda")
        timesteps = torch.randint(0, 1000, (pixel_values.size(0),), device="cuda").long()

        outputs = pipeline.unet(
            pixel_values + noise, timesteps, encoder_hidden_states=input_ids
        )

        # Compute dummy loss (replace with the appropriate diffusion loss)
        loss = loss_fn(outputs.sample, noise)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Batch training complete. Average Loss: {total_loss / len(dataloader)}")

# Main function for batch training
def batch_training(max_batch_size=1000, num_epochs=1):
    """
    Main loop for batch training, iterating through documents in batches.
    """
    print(f"Starting batch training with max batch size: {max_batch_size}")
    optimizer = optim.AdamW(pipeline.unet.parameters(), lr=5e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        while True:
            # Load a batch of documents
            documents = load_batch(max_batch_size)
            if not documents:
                print("No more documents left to train on.")
                break

            # Train on the current batch
            train_on_batch(documents, optimizer, loss_fn)

# Run the batch training process
batch_training(max_batch_size=1000, num_epochs=3)

# Save the fine-tuned model
pipeline.save_pretrained("./fine_tuned_model")
print("Fine-tuned model saved at './fine_tuned_model'")

# Close MongoDB connection
client.close()
print("MongoDB connection closed.")