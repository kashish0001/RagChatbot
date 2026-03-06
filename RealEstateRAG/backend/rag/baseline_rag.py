from backend.retrieval.dense import dense_retriever
from backend.rag.llm import generate_text

def run_baseline_rag(query: str, top_k: int = 3) -> str:
    """
    Broken Baseline RAG implementation.
    Issues demonstrated:
    1. Only uses dense retrieval (misses exact keyword matches).
    2. No chunking strategy or context preservation.
    3. Naive prompt without guardrails (susceptible to prompt injection).
    """
    # 1. Retrieve raw chunks
    results = dense_retriever.search(query, top_k=top_k)
    
    # 2. Naively paste text together
    context = "\n---\n".join([res['text'] for res in results])
    
    # 3. Simple, unprotected prompt
    prompt = f"""
    Answer the user's question using the context.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    answer = generate_text(prompt)
    
    return {
        "answer": answer,
        "retrieved_docs": results,
        "context_used": context
    }
