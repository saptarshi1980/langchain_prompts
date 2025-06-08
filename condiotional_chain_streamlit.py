# app.py
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableMap
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- Models ---
class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment of Feedback")
    highlights: Optional[list[str]] = Field(default=None, description="Key points user liked or disliked")

# --- Parsers ---
parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=FeedBack)

# --- LLM ---
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

# --- Prompt Templates ---
prompt1 = PromptTemplate(
    template="Classify the following feedback into positive or negative sentiment:\n\n{feedback}\n\n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

prompt_positive = PromptTemplate(
    template="""
Write only the body of a formal thank-you letter (no title, no introduction, no commentary) for the following positive feedback. Do not include phrases like "Here is a letter" or "Dear [Name]". 
Start the letter with "Sir/Madam," and end it formally. Organization - THE DURGAPUR PROJECTS LIMITED.

Feedback: "{feedback}"

If any specific points are mentioned below, include them in the letter:
{highlights}
""",
    input_variables=['feedback', 'highlights']
)


prompt_negative = PromptTemplate(
    template="""
Write only the body of a sincere apology letter (no preamble or titles) for the following negative feedback. Do not include phrases like "Here's a letter" or "Dear [Name]". 
Start the letter with "Sir/Madam," and end it formally. Organization - Saptarshi Sanyal.

Feedback: "{feedback}"

If any concerns are mentioned below, include them in the letter:
{highlights}
""",
    input_variables=['feedback', 'highlights']
)

# --- Helper ---
def extract_highlights_safe(x):
    return ", ".join(x.highlights) if x.highlights else " "

# --- Chains ---
chain1 = prompt1 | model | parser2

chain_branch = RunnableBranch(
    (lambda x: x.sentiment == "positive",
     RunnableMap({
         "feedback": lambda x: x,
         "highlights": extract_highlights_safe
     }) | prompt_positive | model | parser),

    (lambda x: x.sentiment == "negative",
     RunnableMap({
         "feedback": lambda x: x,
         "highlights": extract_highlights_safe
     }) | prompt_negative | model | parser),

    RunnableLambda(lambda x: "Could not determine sentiment.")
)

final_chain = chain1 | chain_branch

# --- Streamlit UI ---
st.set_page_config(page_title="Feedback Letter Generator", layout="centered")
st.title("📩 Feedback to Letter Generator")
st.write("Enter your feedback and get a generated letter in response.")

feedback_input = st.text_area("✍️ Enter your feedback here", height=150)

if st.button("Generate Letter") and feedback_input.strip():
    with st.spinner("Generating letter..."):
        try:
            letter = final_chain.invoke(feedback_input.strip())
            #st.success("Letter generated!")
            st.text_area("📄 Generated Letter", value=letter, height=400)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
