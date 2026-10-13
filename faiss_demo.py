from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
docs = [
    Document(page_content="I love programming."),
    Document(page_content="Coding is my passion."),
    Document(page_content="The sun is very hot.")
]

# Embedding function
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS database
vector_db = FAISS.from_documents(docs, embeddings)

print("FAISS database created successfully!")