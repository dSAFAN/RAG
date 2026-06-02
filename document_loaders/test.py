from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
#Reading document
data = TextLoader("document_loaders/Prerequisite.txt")

#Loading Data
docs = data.load()

text_splitter = CharacterTextSplitter(
    separator = "",
    chunk_size = 25,
    chunk_overlap = 10
)

chunks = text_splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()