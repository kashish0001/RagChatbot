from backend.rag.llm import generate_text

def expand_query(original_query: str) -> str:
    """
    Uses an LLM to expand the query with synonyms and related real estate concepts.
    "Is this area safe?" -> "Is this area safe? crime, police reports, safety index, security"
    """
    system_prompt = "You are a Real Estate Query Expansion system. Given a user query, output ONLY 5-10 comma-separated related keywords or concepts that would help retrieve better documents. Do not write anything else. Just the keywords."
    
    prompt = f"Original Query: {original_query}"
    
    keywords = generate_text(prompt, system_prompt=system_prompt)
    
    expanded_query = f"{original_query} {keywords}"
    
    return expanded_query
