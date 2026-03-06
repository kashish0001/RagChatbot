import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class DenseRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Using a fast and small model appropriate for CPU FAISS and standard use cases
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension) # Inner product (Cosine similarities if normalized)
        self.documents = [] # Maps index to document data
        
    def add_documents(self, documents: List[Dict]):
        """
        Embeds and adds documents to the FAISS index.
        documents: [{'id': 1, 'text': '...', 'metadata': {...}}, ...]
        """
        if not documents:
            return
            
        self.documents.extend(documents)
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize for Cosine Similarity inside IndexFlatIP
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform dense vector search using FAISS.
        """
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search returns distances/scores and indices
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                result = self.documents[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
                
        return results

# Singleton instance for easy import
dense_retriever = DenseRetriever()
