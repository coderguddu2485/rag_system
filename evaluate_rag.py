import time
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings

# 1. Updated Test Dataset including Campus, Classrooms, and Sports Locations
TEST_DATASET = [
    {
        "query": "What is the policy for students?",
        "expected_keywords": ["complaint", "attendance", "accounts", "examination"]
    },
    {
        "query": "Where should fee payments be directed?",
        "expected_keywords": ["Accounts Section", "fees", "payment"]
    },
    {
        "query": "Where are classes and academic lectures held at NIT Delhi?",
        "expected_keywords": ["classes", "academic", "lecture", "blocks"]
    },
    {
        "query": "Where is the NIT Delhi permanent campus located?",
        "expected_keywords": ["Narela", "GT Karnal Road", "Delhi", "campus"]
    },
    {
        "query": "What sports grounds and athletic facilities are available?",
        "expected_keywords": ["sports", "ground", "cricket", "football", "basketball"]
    }
]

def evaluate_rag_pipeline():
    print("🚀 Starting RAG Retrieval Evaluation with Campus Queries...\n")
    
    # 2. Setup paths and load vector database
    base_dir = Path(__file__).resolve().parent
    vector_db_dir = base_dir / "vector_db"
    
    embeddings = get_embeddings()
    db = FAISS.load_local(
        str(vector_db_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    total_precision = 0.0
    total_recall = 0.0
    total_latency = 0.0
    num_tests = len(TEST_DATASET)
    
    # 3. Run evaluation loop across all test questions
    for i, item in enumerate(TEST_DATASET, 1):
        query = item["query"]
        keywords = item["expected_keywords"]
        
        # Measure vector search latency
        start_time = time.time()
        docs = db.similarity_search(query, k=3)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        total_latency += latency_ms
        
        # Combine text from retrieved chunks into a single searchable string
        retrieved_text = " ".join([doc.page_content.lower() for doc in docs])
        
        # Evaluate keyword matches (Retrieval Accuracy proxy)
        found_keywords = [kw for kw in keywords if kw.lower() in retrieved_text]
        
        # Calculate metric scores for this query
        precision = len(found_keywords) / len(keywords) if keywords else 0
        recall = len(found_keywords) / len(keywords) if keywords else 0
        
        total_precision += precision
        total_recall += recall
        
        print(f"Test Query {i}: '{query}'")
        print(f"  ⏱️ Latency: {latency_ms:.2f} ms")
        print(f"  🎯 Keywords Found: {len(found_keywords)} / {len(keywords)} {found_keywords}")
        print(f"  📊 Retrieval Score: {precision * 100:.1f}%\n")
        
    # 4. Calculate final averages
    avg_precision = total_precision / num_tests
    avg_recall = total_recall / num_tests
    avg_latency = total_latency / num_tests
    
    print("=" * 50)
    print("📈 FINAL EVALUATION REPORT SUMMARY")
    print("=" * 50)
    print(f"Total Test Cases Evaluated : {num_tests}")
    print(f"Average Vector Search Latency: {avg_latency:.2f} ms")
    print(f"Average Retrieval Precision  : {avg_precision * 100:.1f}%")
    print(f"Average Retrieval Recall     : {avg_recall * 100:.1f}%")
    print("=" * 50)
    print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_rag_pipeline()