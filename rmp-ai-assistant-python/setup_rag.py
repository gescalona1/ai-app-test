from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from open.text.embeddings.openai import OpenAIEmbeddings
import os
import json
from sentence_transformers import SentenceTransformer

print("Loading SentenceTransformer model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("SentenceTransformer model loaded successfully.")

# Load the review data
print("Loading review data from JSON file...")
data = json.load(open("reviews.json"))
print(f"Loaded {len(data['reviews'])} reviews.")

processed_data = []
reviews = [review['review'] for review in data["reviews"]]
print("Encoding reviews...")
embedding_list = model.encode(reviews)
print(f"Encoded {len(embedding_list)} reviews.")

# Create embeddings for each review
print("Processing review data and creating embeddings...")
for i, review in enumerate(data["reviews"]):
    embedding = embedding_list[i].tolist()
    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata":{
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )
print(f"Processed {len(processed_data)} reviews with embeddings.")

# Initialize Pinecone
print("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Pinecone initialized successfully.")

# Create a Pinecone index
print("Creating Pinecone index...")
pc.create_index(
    name="rag",
    dimension=len(embedding),
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print("Pinecone index 'rag' created successfully.")

# Insert the embeddings into the Pinecone index
print("Inserting embeddings into Pinecone index...")
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print("Retrieving index statistics...")
print(index.describe_index_stats())
print("Setup complete.")
