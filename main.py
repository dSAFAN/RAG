from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(model = "mistral-small-2603")

text_data = TextLoader("document_loaders/Prerequisite.txt")
docs_txt = text_data.load()

pdf_data = PyPDFLoader("document_loaders/case_study_2327114.pdf")
docs_pdf = pdf_data.load()

template = ChatPromptTemplate.from_messages(
    [("system","You are an AI Agent that summarizes text and rates the quality of the provided material"),
     ("human","{data}")]
)

final_prompt = template.format_messages(data = docs_pdf)


response = llm.invoke(final_prompt)
print(response.content)
