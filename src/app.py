from loader import load_documents
from splitter import split_documents
from vectordb import get_vector_db
from retriever import retrieve_documents
from prompt import format_context, build_prompt
from llm import generate_answer
import os


def main():
    print("=" * 60)
    print("📄 Welcome to PDF RAG Chatbot (Groq Powered)")
    print("Type 'exit' to quit.")
    print("=" * 60)

    # Step 1 : Load Documents
    documents = load_documents()

    # Step 2 : Split Documents
    chunks = split_documents(documents)

    # Step 3 : Load/Create Vector Database
    vector_db = get_vector_db(chunks)

    while True:
        print("\n" + "-" * 60)

        query = input("Ask Your Question : ").strip()

        if query.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        if not query:
            continue

        # Step 4 : Retrieve Relevant Chunks (returns full documents for metadata access)
        results = retrieve_documents(vector_db, query)

        if not results:
            print("\n🤖 Answer:\nI don't have enough information in the provided documents.")
            continue

        # Step 5 : Convert Chunks to Context
        context = format_context(results)

        # Step 6 : Build Prompt
        prompt = build_prompt(context, query)

        # Step 7 : Generate Answer
        print("\n🤖 Generating answer...")
        answer = generate_answer(prompt)

        # Step 8 : Print Answer
        print(f"\n🤖 Answer:\n{answer}")

        # Step 9 : Print Source Citations
        print("\n📚 Sources:")
        sources = set()
        for doc in results:
            source_path = doc.metadata.get("source", "Unknown File")
            file_name = os.path.basename(source_path)
            page_num = doc.metadata.get("page", 0) + 1  # PDF pages usually start at index 0
            sources.add(f"• {file_name} (Page {page_num})")

        for source in sources:
            print(source)


if __name__ == "__main__":
    main()