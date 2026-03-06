import streamlit as st
import requests
import json
import logging

st.set_page_config(page_title="RealEstateRAG Assistant", page_icon="🏢", layout="wide")

API_URL = "http://localhost:8000"

st.title("🏡 RealEstateRAG: AI-Powered Due Diligence Assistant")
st.markdown("Advanced RAG Multimodal Chatbot demonstrating Hybrid Search, Chunking, Query Expansion, and Vision capabilities.")

# Sidebar for Uploads and Configuration
with st.sidebar:
    st.header("Upload Context")
    
    pdf_file = st.file_uploader("Upload Property Brochure or Legal PDF", type=["pdf"])
    if st.button("Ingest PDF Context Tools"):
        if pdf_file is not None:
            with st.spinner("Chunking and Vectorizing PDF..."):
                files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload/document", files=files)
                if res.status_code == 200:
                    st.success(f"Ingested {res.json().get('chunks')} chunks.")
                else:
                    st.error("Failed to ingest document.")
                    
    img_file = st.file_uploader("Upload Floorplan Image", type=["jpg", "png", "jpeg"])
    if st.button("Analyze & Ingest Floorplan"):
        if img_file is not None:
            with st.spinner("Processing image via BLIP Vision Model..."):
                files = {"file": (img_file.name, img_file.getvalue(), img_file.type)}
                res = requests.post(f"{API_URL}/upload/image", files=files)
                if res.status_code == 200:
                    st.success("Floorplan analyzed!")
                    st.info(f"Generated Caption: {res.json().get('caption')}")
                else:
                    st.error("Failed to process image.")

    st.divider()
    
    st.header("Pipeline Configuration")
    strategy = st.radio("Select RAG Engine:", ["Advanced RAG", "Baseline RAG (Broken)"])
    
    use_hyde = False
    use_qe = True
    
    if strategy == "Advanced RAG":
        st.caption("Advanced features available:")
        use_qe = st.checkbox("Enable Query Expansion", value=True)
        use_hyde = st.checkbox("Enable HyDE (Hypothetical Docs)", value=False)
        
    st.divider()
    if st.button("Run Golden Dataset Evaluation"):
        with st.spinner("Evaluating top queries with Groq LLM-Judge..."):
            res = requests.get(f"{API_URL}/evaluate_golden")
            if res.status_code == 200:
                st.session_state["evaluation_results"] = res.json().get("evaluations", [])
            else:
                st.error("Evaluation failed.")


# Main Chat Interface
query = st.text_input("Ask a question regarding the property (e.g. 'Are there legal risks?' or 'Should I invest?')")

if st.button("Analyze Query"):
    if not query:
        st.warning("Please enter a question.")
    else:
        req_payload = {
            "query": query,
            "strategy": "advanced" if strategy == "Advanced RAG" else "baseline",
            "use_hyde": use_hyde,
            "use_qe": use_qe
        }
        
        with st.spinner("Retrieving and Generating Answer..."):
            try:
                res = requests.post(f"{API_URL}/ask", json=req_payload)
                res.raise_for_status()
                data = res.json()
                
                # Output
                st.subheader("🤖 AI Response")
                st.write(data.get("answer", "No answer generated."))
                
                col1, col2 = st.columns(2)
                with col1:
                    if "investment_score" in data:
                        st.metric(label="Investment Viability Score", value=f"{data['investment_score']}/10")
                        
                with col2:
                    if "transformed_query" in data and data["transformed_query"] != query:
                        with st.expander("Show Internal Transformed/Expanded Query"):
                            st.code(data["transformed_query"])

                # Expandable retrieved chunks
                with st.expander("View Retrieved Context Chunks (Proof of RAG)"):
                    for idx, doc in enumerate(data.get("retrieved_docs", [])):
                        st.markdown(f"**Chunk {idx+1} [Rank: {doc.get('rank', 'N/A')}]**")
                        st.caption(f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
                        st.write(doc.get("text", ""))
                        st.divider()
                        
            except requests.exceptions.HTTPError as e:
                st.error(f"API Error. Please ensure backend is running. ({e})")

# Evaluation display area
if "evaluation_results" in st.session_state:
    st.header("📊 Golden Dataset Evaluation (LLM-as-a-Judge)")
    for ev in st.session_state["evaluation_results"]:
        st.markdown(f"**Query**: {ev['query']} *(Category: {ev['category']})*")
        st.markdown(f"**Judge Score**: {ev['score']}/5")
        st.markdown(f"**Reasoning**: {ev['reasoning']}")
        st.divider()
