import os
import time
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings

def collect_metrics():
    print("📊 Collecting RAG System Technical Metrics...\n")
    
    # 1. Path definitions
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "data" / "NIT_Delhi_RAG_Knowledge_Base.pdf"
    vector_db_dir = base_dir / "vector_db"
    
    # 2. File Size Metrics
    pdf_size_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0
    faiss_size_kb = (vector_db_dir / "index.faiss").stat().st_size / 1024 if (vector_db_dir / "index.faiss").exists() else 0
    pkl_size_kb = (vector_db_dir / "index.pkl").stat().st_size / 1024 if (vector_db_dir / "index.pkl").exists() else 0
    
    print(f"📁 Source PDF Size: {pdf_size_kb:.2f} KB")
    print(f"📁 FAISS Index File Size: {faiss_size_kb:.2f} KB")
    print(f"📁 Pickle Metadata Size: {pkl_size_kb:.2f} KB\n")
    
    # 3. Load Embeddings and Vector DB
    embeddings = get_embeddings()
    db = FAISS.load_local(
        str(vector_db_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Extract chunk metrics directly from the loaded vector store docstore
    docstore_dict = db.docstore._dict
    total_chunks = len(docstore_dict)
    total_chars = sum(len(doc.page_content) for doc in docstore_dict.values())
    avg_chunk_len = total_chars / total_chunks if total_chunks > 0 else 0
    
    print(f"🧩 Total Chunks in Vector DB: {total_chunks}")
    print(f"📏 Average Chunk Length: {avg_chunk_len:.1f} characters\n")
    
    # 4. Embedding Dimensionality Metrics
    sample_vector = embeddings.embed_query("Test query for dimensionality")
    dim = len(sample_vector)
    print(f"📐 Embedding Vector Dimensionality: {dim} dimensions\n")
    
    # 5. Vector Search Latency Benchmark
    print("⏱️ Benchmarking FAISS Vector Search Latency...")
    query = "What is the policy for students?"
    
    # Warm-up run
    _ = db.similarity_search(query, k=3)
    
    # Timed run
    start_time = time.time()
    results = db.similarity_search(query, k=3)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    print(f"⚡ Vector Search Latency: {latency_ms:.2f} ms")
    print(f"🔍 Retrieved Chunks Count: {len(results)}\n")
    
    print("✅ Metrics collection completed successfully!")

if __name__ == "__main__":
    collect_metrics()