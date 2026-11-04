import time
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings

# Define paths
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"

print("📂 Loading Vector Database directly...")

# Load embedding model and FAISS index
embeddings = get_embeddings()
db = FAISS.load_local(
    str(VECTOR_DB_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Database Loaded Successfully!\n")

# Test question for your report metrics
query = "What is the policy for students?"
print(f"🔍 Testing query: '{query}'\n")

# Start timer
start_time = time.time()

# Perform similarity search (retrieve top 3 chunks)
docs = db.similarity_search(query, k=3)

# End timer
end_time = time.time()
latency_ms = (end_time - start_time) * 1000

# Print results
print(f"⏱️ Vector Search Latency: {latency_ms:.2f} ms\n")
print(f"📄 Retrieved {len(docs)} relevant chunks:\n")

for i, doc in enumerate(docs, 1):
    print(f"--- Chunk {i} ---")
    snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
    print(snippet)
    print()