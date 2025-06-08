from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq


load_dotenv()
api_key= os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API Key not found in .env file")

promtpt1= PromptTemplate(
    template="Generate a detail report on the following topic \n {topic}",
    input_variable=["topic"]
)

promtpt2 = PromptTemplate(
    template="Give a 5 pointer summary on the flllowing text \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

model=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key    
)

chain = promtpt1|model|parser|promtpt2|model|parser
result = chain.invoke({"topic":"Indian Premiur League"})
print(result)
chain.get_graph().print_ascii();

