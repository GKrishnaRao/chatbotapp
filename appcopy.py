import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the Groq and Google Generative AI API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Chat with Real AI")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided DCPR 2034 documents only.
Please provide the most accurate response based on the question.
Context: {context}
Question: {input}
""")


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        

prompt1 = st.text_input("Ask questions related to DCPR 2034")


if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is Ready")
    

import time

if prompt1:
    start_time = time.time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response["answer"])
    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"Execution time: {execution_time} seconds")   
    
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")