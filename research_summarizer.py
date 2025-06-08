from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization (reduces RAM usage)
)

# Load environment variables (if using HF token)
load_dotenv()

# Initialize the Hugging Face pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
       "temperature": 0.1,  # Lower = more factual
    "do_sample": True,
    "repetition_penalty": 1.2,  # Reduce nonsense repeats
    "max_new_tokens": 200,
       "max_new_tokens": 500,
        "device_map": "cpu"  # Use "cpu" if no GPU
    }
)

model = ChatHuggingFace(llm=llm)

# Define the prompt template
template = """
Summarize the paper '{paper_input}' in a {style_input} style.
Length: {length_input}.
Provide key insights and methodologies clearly.
"""

prompt = PromptTemplate.from_template(template)

# User inputs (hardcoded for testing)
paper_input = "Attention Is All You Need"  # Try: "BERT", "GPT-3", etc.
style_input = "Technical"                 # Try: "Beginner-Friendly", "Mathematical"
length_input = "Medium"                   # Try: "Short", "Long"

# Generate and print the summary
print("\nGenerating summary...")
chain = prompt | model
result = chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

print("\n=== Summary ===")
print(result.content)