from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
#Reading document
data = PyPDFLoader("document_loaders/case_study_2327114.pdf")

'''
text_splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

'''
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
#Loading Data
docs = data.load()
chunks = text_splitter.split_documents(docs)

print(chunks[5].page_content)