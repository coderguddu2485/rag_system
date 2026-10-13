def format_context(results):
    """
    Convert retrieved documents into one context string.
    """
    context = "\n\n".join(
        doc.page_content for doc in results
    )
    return context


def build_prompt(context, query):
    """
    Build a flexible, intelligent prompt for the LLM that handles partial matches 
    and provides helpful general responses instead of strict refusals.
    """
    prompt = f"""You are an expert AI assistant for a college/institution knowledge system. 

Follow these guidelines when answering the user:
1. **Primary Source**: Use the provided context below as your primary reference point.
2. **Partial Context (30-40% Match)**: If the context contains partial, related, or indirect details about the question, use them as clues and smoothly combine them with your own general knowledge to construct a complete, helpful answer.
3. **Missing Information**: If the exact details are not found in the context, DO NOT give a rigid or dry refusal. Instead, provide a helpful and informative response using your general knowledge, while politely adding context that the specific institutional document did not outline those exact details.

Context:
{context}

Question: {query}

Answer:"""
    return prompt