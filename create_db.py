from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

# Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

# Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)


# Pdf Loader
pdf_data = PyPDFLoader("document_loaders/clean-code.pdf")
docs_pdf = pdf_data.load()

# Chunk creation
chunks = splitter.split_documents(docs_pdf)

# embedding and db creation
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    persist_directory = "chromaDB"
)
