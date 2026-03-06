from backend.rag.llm import generate_text

def generate_hyde_document(query: str) -> str:
    """
    Hypothetical Document Embeddings (HyDE) strategy.
    Generates a hypothetical ideal response to the user's real estate query.
    This generated text is then embedded to find semantically similar real documents.
    """
    system_prompt = """You are a highly knowledgeable Real Estate Agent and Property Inspector.
    Write a 3-4 sentence hypothetical and perfect factual answer to the given query. 
    Write as if you are quoting directly from a property brochure or legal document.
    Do not add conversational fluff. Just produce the hypothetical document passage."""
    
    hypothetical_doc = generate_text(query, system_prompt=system_prompt)
    
    return hypothetical_doc
