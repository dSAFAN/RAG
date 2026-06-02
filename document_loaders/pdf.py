from langchain_community.document_loaders import PyPDFLoader

#Reading document
data = PyPDFLoader("document_loaders/case_study_2327114.pdf")

#Loading Data
docs = data.load()

print(docs[1])