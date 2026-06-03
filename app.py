import streamlit as st
import os
import tempfile
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# 1. Initialization
load_dotenv()
st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚")
st.title("📚 Chat with your PDF")

# Initialize models (Do this once)
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2603")

embedding = get_embedding_model()
llm = get_llm()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Prompt Template
template = ChatPromptTemplate.from_messages([
    ("system",
     """You are a Helpful AI Assistant.
        Use ONLY the provided context to answer the question.
        If the answer is not present in the context, say "I Could not find the answer in the document."
     """),
    ("human",
     """Context : {context}
        Question : {question}
     """)
])

# 2. Sidebar: File Upload & Database Creation (From create_db.py)
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to chat with", type="pdf")

    if uploaded_file is not None and st.button("Process Document"):
        with st.spinner("Processing PDF..."):
            # Streamlit uploads are kept in memory. We need to save it to a temporary file 
            # so PyPDFLoader can read it from a file path.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Load and Split
                pdf_data = PyPDFLoader(tmp_file_path)
                docs_pdf = pdf_data.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs_pdf)

                # Create Vector Store in memory (no need to save to disk for a quick chat UI)
                # By not providing a persist_directory, it stays ephemeral.
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding
                )
                st.success("Document processed successfully! You can now chat.")
                
                # Clear previous chat history when a new document is uploaded
                st.session_state.messages = []

            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Clean up the temporary file
                os.remove(tmp_file_path)

# 3. Main Chat Interface (From main.py)
st.header("2. Chat Interface")

# Display Chat History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle New User Input
if prompt := st.chat_input("Ask a question about your PDF..."):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF document first.")
    else:
        # Display user message instantly
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Retrieve and Generate
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                # Setup Retriever
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
                )
                
                # Fetch Context
                docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Generate Response
                final_prompt = template.invoke({"question": prompt, "context": context})
                response = llm.invoke(final_prompt)
                
                st.markdown(response.content)
                
                # Expandable UI element to show the user what chunks were retrieved!
                with st.expander("View Retrieved Context Chunks"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.caption(doc.page_content)
        
        # Save AI response to history
        st.session_state.messages.append(AIMessage(content=response.content))