def retrieve_documents(vector_db, query, k=3):
    """
    Retrieve the top-k most relevant documents.
    """

    results = vector_db.similarity_search(
        query=query,
        k=k
    )

    return results