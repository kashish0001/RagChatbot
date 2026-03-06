import sys
import os
import json
from pathlib import Path

# Add project root to sys.path so we can import backend packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.rag.advanced_rag import run_advanced_rag
from backend.evaluation.llm_judge import evaluate_answer

def calculate_metrics(query, expected_concepts, retrieved_docs):
    """
    Calculate Precision, Recall, and MRR based on expected_concepts presence in retrieved_docs.
    """
    if not expected_concepts:
         return 0.0, 0.0, 0.0
         
    expected_lower = [c.lower() for c in expected_concepts]
    
    # Check which expected concepts are found anywhere in the retrieved docs
    found_concepts = set()
    
    relevant_docs_count = 0
    mrr = 0.0
    
    for rank, doc in enumerate(retrieved_docs, start=1):
        text = doc.get("text", "").lower()
        is_relevant = False
        
        for concept in expected_lower:
            if concept in text:
                found_concepts.add(concept)
                is_relevant = True
                
        if is_relevant:
            relevant_docs_count += 1
            if mrr == 0.0:
                 mrr = 1.0 / rank
                 
    # Precision: Relevant docs / Total docs retrieved
    total_docs = len(retrieved_docs)
    precision = relevant_docs_count / total_docs if total_docs > 0 else 0.0
    
    # Recall: Unique expected concepts found / Total expected concepts length
    recall = len(found_concepts) / len(expected_concepts)
    
    return precision, recall, mrr

def main():
    print("Starting CLI Evaluation for RealEstateRAG...")
    
    filepath = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    if not os.path.exists(filepath):
        print(f"Error: dataset not found at {filepath}")
        return
        
    with open(filepath, "r") as f:
        dataset = json.load(f)
        
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    total_score = 0.0
    
    num_queries = len(dataset)
    
    for idx, item in enumerate(dataset, start=1):
        query = item['query']
        expected = item['expected_answer_concepts']
        category = item['category']
        
        print(f"\n--- Evaluating Query {idx}/{num_queries} ---")
        print(f"Query: {query}")
        print(f"Category: {category}")
        
        # Run advanced RAG pipeline
        rag_output = run_advanced_rag(query, use_hyde=False, use_qe=True)
        answer = rag_output['answer']
        retrieved_docs = rag_output['retrieved_docs']
        
        # Calculate retrieval metrics
        precision, recall, mrr = calculate_metrics(query, expected, retrieved_docs)
        
        # Get LLM Judge Score
        eval_result = evaluate_answer(query, answer, expected)
        score = eval_result.get('score', 0)
        reasoning = eval_result.get('reasoning', '')
        
        print(f"LLM Judge Score: {score}/5")
        if score == 0:
             print(f"Reasoning/Error: {reasoning}")
             
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"MRR: {mrr:.4f}")
        
        total_precision += precision
        total_recall += recall
        total_mrr += mrr
        total_score += score
        
    if num_queries > 0:
        print("\n==============================")
        print("AGGREGATE METRICS")
        print("==============================")
        print(f"Average LLM Judge Score: {total_score / num_queries:.2f}/5")
        print(f"Average Precision:       {total_precision / num_queries:.4f}")
        print(f"Average Recall:          {total_recall / num_queries:.4f}")
        print(f"Average MRR:             {total_mrr / num_queries:.4f}")

if __name__ == "__main__":
    main()
