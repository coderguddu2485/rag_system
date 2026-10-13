from pathlib import Path

from langchain_community.vectorstores import FAISS

# from embeddings import get_embeddings
from src.embeddings import get_embeddings

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder where FAISS index will be stored
VECTOR_DB_DIR = BASE_DIR / "vector_db"


def create_vector_db(chunks):
    """
    Create a FAISS vector database from document chunks
    and save it to disk.
    """

    # Create vector_db folder if it doesn't exist
    VECTOR_DB_DIR.mkdir(exist_ok=True)

    # Load embedding model
    embeddings = get_embeddings()

    # Create FAISS database
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save FAISS database
    vector_db.save_local(str(VECTOR_DB_DIR))

    print("✅ Vector Database Created Successfully!")

    return vector_db


def load_vector_db():
    """
    Load an existing FAISS vector database.
    """

    embeddings = get_embeddings()

    vector_db = FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ Vector Database Loaded Successfully!")

    return vector_db


def get_vector_db(chunks):
    """
    If vector database already exists, load it.
    Otherwise create a new one.
    """

    index_file = VECTOR_DB_DIR / "index.faiss"

    if index_file.exists():
        print("📂 Loading Existing Vector Database...")
        return load_vector_db()

    print("🆕 Creating New Vector Database...")
    return create_vector_db(chunks)