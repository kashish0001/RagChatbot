import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "mistral" # Can also use 'llama3' depending on developer setup

def generate_text(prompt: str, system_prompt: str = "") -> str:
    """
    Submits a prompt to the local Ollama LLM API and returns the response.
    """
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    if system_prompt:
        payload["system"] = system_prompt
        
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "Sorry, there was an error communicating with the local language model. Please make sure Ollama is running."
