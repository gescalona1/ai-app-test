from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from open.text.embeddings.openai import OpenAIEmbeddings
import os
import json


# Load the review data
data = json.load(open("reviews.json"))

processed_data = []
embeddings = OpenAIEmbeddings(
    openai_api_base='https://limcheekin-bge-small-en-v1-5.hf.space/v1',
    openai_api_key='Your HuggingFace Token' # this actually doens't do anything
)

# Create embeddings for each review
for review in data["reviews"]:
    embedding = embeddings.embed_query(review['review'])
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
