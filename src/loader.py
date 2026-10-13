from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Folder
DATA_DIR = BASE_DIR / "data"


def load_documents():
    """
    Load all PDF files from the data folder.

    Returns:
        list: List of LangChain Document objects.
    """

    documents = []

    # Find all PDF files
    pdf_files = DATA_DIR.glob("*.pdf")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents