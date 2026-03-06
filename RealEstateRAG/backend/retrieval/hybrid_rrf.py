from typing import List, Dict

def reciprocal_rank_fusion(
    bm25_results: List[Dict], 
    dense_results: List[Dict], 
    k: int = 60, 
    top_n: int = 5
) -> List[Dict]:
    """
    Combines BM25 and Dense Retrieval results using Reciprocal Rank Fusion (RRF).
    RRF Score = sum( 1 / (k + rank) ) across all retrieval methods.
    """
    
    rrf_scores = {}
    doc_lookup = {}
    
    # Process BM25
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.get('id', doc['text']) # fallback to text if id is missing
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            doc_lookup[doc_id] = doc
        rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
    # Process Dense
    for rank, doc in enumerate(dense_results):
        doc_id = doc.get('id', doc['text'])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            doc_lookup[doc_id] = doc
        rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
    # Sort by RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for rank, (doc_id, score) in enumerate(sorted_docs[:top_n]):
        result = doc_lookup[doc_id].copy()
        result['rrf_score'] = score
        result['final_rank'] = rank + 1
        final_results.append(result)
        
    return final_results
