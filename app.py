import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Resume AI Assistant", page_icon="🤖")
st.title("📄 Chat with my Resume")

# 🔑 API Key
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# 📤 Upload PDF
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

@st.cache_resource
def initialize_rag(file_bytes):
    with open("resume.pdf", "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

if uploaded_file and openai_api_key:
    try:
        vectorstore = initialize_rag(uploaded_file.read())
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

       # Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask about your resume...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        docs = retriever.invoke(user_input)

        context = "\n\n".join([doc.page_content for doc in docs])

        response = llm.invoke(f"""
Answer based on this resume:

{context}

Question: {user_input}
""").content

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a resume and enter API key to start.")
