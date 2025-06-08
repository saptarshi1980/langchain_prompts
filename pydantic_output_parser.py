from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# ---- Step 1: Define Pydantic model ----
class PersonInfo(BaseModel):
    name: str = Field(description="The full name of the person")
    age: int = Field(description="The age of the person")

# ---- Step 2: Setup output parser ----
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# ---- Step 3: Prompt with formatting instructions ----
prompt = PromptTemplate(
    template="Extract name and age from this text:\n{text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ---- Step 4: Load API key and LLM ----
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=api_key
)

# ---- Step 5: Chain together ----
chain = prompt | llm | parser

# ---- Step 6: Call chain ----
input_text = "My name is Saptarshi Sanyal and I am 45 years old."
result = chain.invoke({"text": input_text})

# ---- Step 7: Output as Python object ----
print("Structured Output:")
print(result)
print("Name:", result.name)
print("Age:", result.age)
