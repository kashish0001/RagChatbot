from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
#import spacy # Note: Spacy might be needed if complex tokenization, but we use simple regex in bm25. 
import PyPDF2
from PIL import Image
import io
import uuid

# Import modules from our system
from backend.retrieval.bm25 import bm25_retriever
from backend.retrieval.dense import dense_retriever
from backend.retrieval.parent_child_chunking import chunker
from backend.rag.baseline_rag import run_baseline_rag
from backend.rag.advanced_rag import run_advanced_rag
from backend.vision.floorplan_analyzer import analyzer as vision_analyzer
# Additional imports for evaluation
import json
import os
from backend.evaluation.llm_judge import evaluate_answer

app = FastAPI(title="RealEstateRAG Context API")

class QueryRequest(BaseModel):
    query: str
    strategy: str = "advanced" # 'baseline', 'advanced'
    use_hyde: bool = False
    use_qe: bool = True

@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests PDF documents, processes them via parent-child chunking, 
    and adds them to both dense and sparse indices.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    pdf_reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n\n"
        
    metadata = {
        "source": file.filename,
        "type": "document"
    }
    
    # Process text using parent-child chunking natively
    child_docs = chunker.chunk_document(text, source_metadata=metadata)
    
    if not child_docs:
         return {"message": "Document uploaded but no recognizable text was extracted."}
         
    # Add identical documents to both retrievers
    bm25_retriever.add_documents(child_docs)
    dense_retriever.add_documents(child_docs)
    
    return {"message": f"Successfully ingested {file.filename} into indices.", "chunks": len(child_docs)}

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts an Image, captions it with BLIP, and treats the caption as a context document.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid Image File")
        
    caption = vision_analyzer.analyze_image(image)
    
    doc_id = str(uuid.uuid4())
    img_doc = [{
        "id": doc_id,
        "text": f"Image Description of {file.filename}: {caption}",
        "metadata": {
            "source": file.filename,
            "type": "image",
            # Add parent_id identically so the chunker resolver doesn't throw errors
            "parent_id": doc_id 
        }
    }]
    
    # Add the single image sentence to the index. 
    # Also add it to chunker.parents so it can be resolved natively by the chunker.
    chunker.parents[doc_id] = f"Image Description of {file.filename}: {caption}"
    
    bm25_retriever.add_documents(img_doc)
    dense_retriever.add_documents(img_doc)
    
    return {"message": "Image analyzed and added to context.", "caption": caption}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Main endpoint for answering user's Real Estate queries.
    Routes to the specified RAG strategy.
    """
    # Guard against empty queries or indices, though baseline_rag handles empty somewhat.
    if request.strategy == "baseline":
        return run_baseline_rag(request.query)
        
    elif request.strategy == "advanced":
        return run_advanced_rag(
            request.query, 
            use_hyde=request.use_hyde, 
            use_qe=request.use_qe
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy specified.")

@app.get("/evaluate_golden")
async def run_evaluation():
    """
    Loads golden_dataset.json, runs queries through advanced RAG, and evaluates them with LLM Judge.
    """
    filepath = os.path.join(os.path.dirname(__file__), "evaluation", "golden_dataset.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Golden dataset not found.")
        
    with open(filepath, "r") as f:
        dataset = json.load(f)
        
    results = []
    # Using top 3 queries for quick evaluation demonstration
    for item in dataset[:3]: 
        query = item['query']
        expected = item['expected_answer_concepts']
        
        # We enforce advanced pipeline for evaluation
        rag_output = run_advanced_rag(query, use_hyde=False, use_qe=True)
        answer = rag_output['answer']
        
        eval_result = evaluate_answer(query, answer, expected)
        
        results.append({
            "query": query,
            "category": item['category'],
            "score": eval_result.get('score', 0),
            "reasoning": eval_result.get('reasoning', '')
        })
        
    return {"evaluations": results}
