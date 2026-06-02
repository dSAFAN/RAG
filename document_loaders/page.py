from langchain_community.document_loaders import WebBaseLoader

url = "https://techsy.io/en/blog/best-rag-tools"

data = WebBaseLoader(url)
docs = data.load()
print(docs[0].page_content)