from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq


# I am loading the envrionment variable for using GROQ API key here.
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Raising an exception if the no value for the GROQ API is found the .env file.
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Check your .env file.")

#I am designing a prompt template here. I want to make it generic so that i can be able to generate 5 interesting facts on any topic which is enetered by user at run time and i will substitue that topic at runtime. 
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

#Creating the the LLM model instance here
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

#I want to structure the output given my the LLM. I am using output parser
parser = StrOutputParser()

#I am creating a chain here, first the prompt will be formed, that will be go as input to the LLM. The output of LLM will go to parserr as input and finally the parser will give the output of the chain.
chain = prompt | model | parser

# I am calling invoke method of the chain, not of the model. This the beauty. I am providing the topic here. It is about making the application dynamic.
result = chain.invoke({'topic':'cricket'})

#Printing the result
print(result)

#Printing the flow of control of the program
chain.get_graph().print_ascii()