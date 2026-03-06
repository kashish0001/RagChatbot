import re
from rank_bm25 import BM25Okapi
from typing import List, Dict

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []  # List of dicts with 'id', 'text', 'metadata'
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace and punctuation tokenizer
        text = text.lower()
        return re.findall(r'\w+', text)

    def add_documents(self, documents: List[Dict]):
        """
        Adds documents to the BM25 index.
        documents: [{'id': 1, 'text': '...', 'metadata': {...}}, ...]
        """
        self.documents = documents
        tokenized_corpus = [self._tokenize(doc['text']) for doc in documents]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform BM25 search and return top_k documents.
        """
        if not self.bm25 or not self.documents:
            return []
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Pair score with document index
        scored_docs = [(score, i) for i, score in enumerate(doc_scores)]
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scored_docs[:top_k]:
            if score > 0: # Only return if there is some match
                result = self.documents[idx].copy()
                result['score'] = score
                result['rank'] = len(results) + 1
                results.append(result)
                
        return results

# Example usage singleton
bm25_retriever = BM25Retriever()
