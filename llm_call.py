from text_extractor import TextExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity
from dotenv import load_dotenv
import os

load_dotenv()  

from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

chat = ChatPerplexity(temperature=0, pplx_api_key=os.environ["PPLX_API_KEY"], model="sonar")

system = (
    "You are a helpful assistant that reads and summarizes academic research papers "
    "in a way that is accessible to a general audience. You extract key sections and explain technical content in simple terms."
)

human = """
Please summarize the following research paper content clearly and concisely. 
Break the summary into these sections:

1. **Introduction / Abstract** – What is the paper about? What problem does it aim to solve?
2. **Methodology** – How was the research conducted? Briefly explain the approach or experiment.
3. **Theory / Mathematics** – Explain any important theoretical background or equations used in simple terms.
4. **Key Diagrams or Visual Elements** – If the paper includes diagrams, describe what they represent and their purpose.
5. **Conclusion** – What are the key takeaways? What did the paper achieve or suggest for the future?

Ensure the summary is easy to understand, even for someone without a technical background.

Text to summarize:
{input}
"""

pdf_path = "/home/hari/computer_science/ai_projects/AI_Paper_Summarizer/Underwater Image Enhancement Using FPGA-Based Gaussian Filters.pdf"
extractor = TextExtractor(pdf_path)
paper_text = extractor.extract_text_from_pdf()

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

response = chain.invoke({"input": paper_text})
print(response.content)