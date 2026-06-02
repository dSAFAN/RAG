from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(
    model = "mistral-small-2603"
)

response = llm.invoke("Hey, you there?")
print(response)
