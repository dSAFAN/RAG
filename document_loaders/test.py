from langchain_community.document_loaders import TextLoader

#Reading document
data = TextLoader("document_loaders/Prerequisite.txt")

#Loading Data
docs = data.load()

print(docs[0].page_content)