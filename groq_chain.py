from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# ---- Load Environment Variables from .env ----
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Check your .env file.")

# ---- Prompt Template ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent and friendly assistant. Answer clearly and concisely."),
    ("user", "{question}")
])

# ---- Groq LLM ----
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

# ---- Chain (Prompt ➜ LLM) ----
chain: RunnableSequence = prompt | llm

# ---- Chat Loop ----
print("🤖 General Q&A ChatBot (Groq + LangChain)\nType 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ")
    if user_input.strip().lower() in {"exit", "quit", "bye"}:
        print("👋 Goodbye!")
        break
    try:
        response = chain.invoke({"question": user_input})
        print("🤖", response.content.strip(), "\n")
    except Exception as e:
        print("⚠️ Error:", str(e))
