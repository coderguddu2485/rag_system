#    reading the file from the data directory and printing the first page of the first readable PDF file

# from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader

# pdf_dir = Path("data")
# pdf_files = sorted(pdf_dir.glob("*.pdf"))

# if not pdf_files:
#     print("No PDF files found in the data directory.")
#     raise SystemExit(0)

# for pdf_path in pdf_files:
#     try:
#         loader = PyPDFLoader(str(pdf_path))
#         docs = loader.load()
#         if docs:
#             print(docs[0].page_content)
#             raise SystemExit(0)
#     except Exception as exc:
#         print(f"Skipping unreadable PDF {pdf_path.name}: {exc}")

# print("No readable PDF files were found in the data directory.")
# raise SystemExit(0)


#                   chunking the first page of the first readable PDF file into smaller chunks and printing them

# from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# pdf_path = Path("data/guddu_resume1.pdf")

# loader = PyPDFLoader(str(pdf_path))

# docs = loader.load()

# print("Total Pages:", len(docs))

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=50
# )

# chunks = splitter.split_documents(docs)

# print("Total Chunks:", len(chunks))

# print("\nFirst Chunk:\n")
# print(chunks[0].page_content)
# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i + 1}:\n")
#     print(chunk.page_content)


