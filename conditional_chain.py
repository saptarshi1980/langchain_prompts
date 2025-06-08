from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal,Optional
from langchain.schema.runnable import RunnableMap

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

parser = StrOutputParser()


class FeedBack(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="Sentiment of Feedback")
    highlights:Optional[list[str]]=Field(default='None',description="key points user liked or disliked")

parser2=PydanticOutputParser(pydantic_object=FeedBack)

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
)

prompt1 = PromptTemplate(
    template="Classify the following feedback into postive or negative sentiment \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)



prompt_positive = PromptTemplate(
    template="""Write a professional thank-you letter for the following positive feedback. Avoid spaceholder. Start with Sir/Madam. Organization - The Durgapur Projects Limited

Feedback: "{feedback}"

If any specific points are mentioned below, include them in the letter:
{highlights}
""",
    input_variables=['feedback', 'highlights']
)

prompt_negative = PromptTemplate(
    template="""Write a sincere apology letter for the following negative feedback.Avoid spaceholder. Start with Sir/Madam. Organization - The Durgapur Projects Limited

Feedback: "{feedback}"

If any concerns are mentioned below, include them in the letter:
{highlights}
""",
    input_variables=['feedback', 'highlights']
)


chain1=prompt1|model|parser2

feedback_str='Hanging issue, its not working like 4 gb ram mobile it Work like a 2 gb ram Mobile, worst experience.'
#feedback_str='This is a beatiful hotel, the stay was amazing'


def extract_highlights_safe(x):
    return " ,".join(x.highlights) if x.highlights else " "
 
    
chain_branch = RunnableBranch(
    
    (lambda x:x.sentiment=="positive", RunnableMap({
         "feedback": lambda x: x,
         "highlights": extract_highlights_safe
     }) |prompt_positive|model|parser),
    (lambda x:x.sentiment=="negative",RunnableMap({
         "feedback": lambda x: x,
         "highlights": extract_highlights_safe
     }) |prompt_negative|model|parser),
    RunnableLambda(lambda x:'Could not determine sentiment')
    
)

final_chain= chain1|chain_branch

result = final_chain.invoke(feedback_str)

print(result)

#chain1.get_graph().print_ascii()

