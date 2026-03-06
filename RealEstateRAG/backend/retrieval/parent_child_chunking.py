import re
import uuid
from typing import List, Dict

class ParentChildChunker:
    """
    Implements Parent-Child chunking architecture.
    Parent = Larger context (e.g., paragraph or section), 
    Child = Smaller, more specific retrieval unit (e.g., sentence).
    The retrieval mechanism finds the closest child chunk, but returns the linked parent.
    """
    def __init__(self):
        self.parents = {} # parent_id -> parent_text
        self.child_documents = [] # List of child dicts suitable for dense/BM25 retrievers
        
    def chunk_document(self, text: str, source_metadata: Dict = None) -> List[Dict]:
        """
        Splits a document text into parents (paragraphs) and children (sentences).
        """
        if source_metadata is None:
            source_metadata = {}
            
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20: continue # Skip very small blocks
            
            parent_id = str(uuid.uuid4())
            self.parents[parent_id] = paragraph
            
            # Simple sentence splitting for children
            # Using regex to split by ., !, or ? followed by space and uppercase letter.
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 5: continue
                
                child_id = str(uuid.uuid4())
                
                # We store parent ID in the child metadata
                meta = source_metadata.copy()
                meta['parent_id'] = parent_id
                
                self.child_documents.append({
                    'id': child_id,
                    'text': sentence,
                    'metadata': meta
                })
                
        return self.child_documents

    def resolve_parents(self, child_results: List[Dict]) -> List[Dict]:
        """
        Given a list of retrieved child chunks, fetch their respective parent chunks.
        Deduplicates if multiple child chunks point to the same parent.
        """
        resolved_parents = []
        seen_parents = set()
        
        for child in child_results:
            parent_id = child.get('metadata', {}).get('parent_id')
            
            if parent_id and parent_id not in seen_parents and parent_id in self.parents:
                parent_text = self.parents[parent_id]
                seen_parents.add(parent_id)
                
                # Return the parent document, preserving the child's score for ranking
                parent_doc = {
                    'id': parent_id,
                    'text': parent_text,
                    'metadata': child.get('metadata', {}),
                    'child_trigger': child['text'],
                    'score': child.get('score', 0),
                    'rank': len(resolved_parents) + 1
                }
                resolved_parents.append(parent_doc)
                
        return resolved_parents

chunker = ParentChildChunker()
