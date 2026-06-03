from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import chroma 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)


pdf_data = PyPDFLoader("document_loaders/case_study_2327114.pdf")
docs_pdf = pdf_data.load()
chunks = splitter.split_documents(docs_pdf)

print(len(chunks))