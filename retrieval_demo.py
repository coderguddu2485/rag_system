from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content="Database normalization reduces redundancy."),
    Document(page_content="Primary key uniquely identifies each record."),
    Document(page_content="TCP is a connection-oriented protocol."),
    Document(page_content="React is a frontend JavaScript library.")
]

vector_db = FAISS.from_documents(docs, embeddings)

query = "Explain normalization"

results = vector_db.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}")
    print(doc.page_content)
print(type(vector_db))
print(vector_db.index)