from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(model = "mistral-small-2603")
data = TextLoader("document_loaders/Prerequisite.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system","You are an AI Agent that summarizes any given data"),
     ("human","{data}")]
)

final_prompt = template.format_messages(data = docs[0].page_content)


response = llm.invoke(final_prompt)
print(response.content)
