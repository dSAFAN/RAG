from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Embedding Model
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Load DB
vectorstore = Chroma(
    persist_directory = "chromaDB",
    embedding_function = embedding
)

# Retriever Creation - MMR
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k" : 4,
                     "fetch_k" : 10,
                     "lambda_mult" : 0.5}
)


# Model instantiate
llm = ChatMistralAI(model = "mistral-small-2603")

# Prompt Template 
template = ChatPromptTemplate.from_messages(
    [("system",
      """ You are a Helpful AI Assistant.
          Use ONLY the provided context to answer the question.
          If the answer is not present in the context, say "I Could not find the answer in the document."
      """),
     ("human",
      """ Context : {context}
          Question : {question}
      """)
      ])

print("RAG System Created")
print("Press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break
    
    # Converting user question in embeddings and doing a mmr search in vector db
    # returns similar chunks
    docs = retriever.invoke(query)

    # joining similar chunks for LLM Model
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = template.invoke({"question" : query, "context" : context})

    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}" )