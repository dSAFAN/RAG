from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
#Reading document
data = PyPDFLoader("document_loaders/case_study_2327114.pdf")
text_splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)
#Loading Data
docs = data.load()
chunks = text_splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print("-------------")
    print("-------------")