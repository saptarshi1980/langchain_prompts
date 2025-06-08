from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import torch

# Load environment variables
load_dotenv()

# Configuration - Using a smaller CPU-friendly model
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # 4-bit quantized (efficient)

# Custom download location on D: drive
MODELS_DIR = "D:/ai_models/mistral"  # Use forward slashes or double backslashes

def load_model():
    """Load the quantized GGUF model for CPU from custom location"""
    from llama_cpp import Llama
    
    # Create directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Full model path
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    
    # Download model if not present
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILE,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False  # Avoids symlinks for better portability
        )
        print("Download complete!")
    
    # Initialize LLM
    print(f"Loading model from {model_path}...")
    return Llama(
        model_path=model_path,
        n_ctx=2048,      # Context window
        n_threads=4,     # CPU threads to use
        n_gpu_layers=0   # CPU-only mode
    )

def generate_summary(paper_title):
    """Generate a factual summary with verification"""
    llm = load_model()
    
    # Carefully structured prompt
    prompt = f"""<|system|>
You are an AI research assistant. Summarize this paper accurately.
Key requirements:
1. Only use information from the actual paper
2. Must mention "Vaswani et al." as authors
3. Focus on technical contributions
4. Never invent details</|system|>
<|user|>
Paper: {paper_title}
Please provide a 3-paragraph technical summary covering:
- The Transformer architecture
- Self-attention mechanism
- Key results from the WMT 2014 evaluation</|user|>
"""
    
    # Generate with constraints
    output = llm(
        prompt,
        max_tokens=350,
        temperature=0.1,  # Low for factual accuracy
        top_p=0.9,
        echo=False
    )
    
    return output['choices'][0]['text']

if __name__ == "__main__":
    try:
        paper = "Attention Is All You Need"
        print(f"\nGenerating summary for: {paper}")
        
        summary = generate_summary(paper)
        
        # Basic verification
        if "Vaswani" not in summary:
            summary = "[VERIFICATION WARNING] " + summary
        
        print("\n=== Research Summary ===")
        print(summary)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print(f"- Check write permissions for {MODELS_DIR}")
        print("- Ensure you have 5GB+ free disk space on D:")
        print("- Try reducing n_ctx to 1024 if memory constrained")