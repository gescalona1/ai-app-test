from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from open.text.embeddings.openai import OpenAIEmbeddings
import os
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load the review data
data = json.load(open("reviews.json"))

processed_data = []
reviews = [review['review'] for review in data["reviews"]]
embedding_list = model.encode(reviews)
# Create embeddings for each review
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
print(processed_data)
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a Pinecone index
pc.create_index(
    name="rag",
    dimension=len(embedding),
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)


# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())
