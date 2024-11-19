import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
#from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

# Load the Groq and Google Generative AI API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Chat with Real AI")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided DCPR 2034 documents only.
Please provide the most accurate response based on the question.

Context: {context}
Question: {input}

Answer:
""")

# Create the document chain with output key
document_chain = create_stuff_documents_chain(
    llm, 
    prompt,
    document_variable_name="context",
)


def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Create or get the index
        index_name = "realincgemma"
        dimension = 768  # For Google Generative AI Embeddings


        # Check if index exists, if not create it
        if index_name not in pc.list_indexes().names():
            print(index_name)
        
        # Get the index
        index = pc.Index(index_name)   
        print(index)
            
            
        # Create embeddings      
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load and process documents
        loader = PyPDFDirectoryLoader("pdf")
        docs = loader.load()
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)
        print(documents)
        print('============ Outside the Code ============')
        # Create vectors and store in Pinecone
        for i, doc in enumerate(documents):
            vector = embeddings.embed_query(doc.page_content)
            metadata = {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0)
            }
            print('============ Run the Code ============ ', str(i))
            index.upsert(vectors=[{
                "id": f"doc_regulation{i}",
                "values": vector,
                "metadata": metadata
            }])
            
        st.session_state.index = index
        st.session_state.embeddings = embeddings    
        #st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


def similarity_search(query, k=3):
    # Generate embedding for the query
    query_embedding = st.session_state.embeddings.embed_query(query)
    
    # Perform similarity search
    results = st.session_state.index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    
    # Convert results to document format
    documents = []
    for match in results['matches']:
        doc = Document(
            page_content=match['metadata']['text'],
            metadata={
                'source': match['metadata']['source'],
                'page': match['metadata']['page']
            }
        )
        documents.append(doc)
    
    return documents      

prompt1 = st.text_input("Ask questions related to DCPR 2034")


if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is Ready")
    

import time

if prompt1:
    start_time = time.time()
    #document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Use Pinecone for similarity search
    similar_docs = similarity_search(prompt1)
    
    # Create retrieval chain with the similar documents
    response = document_chain.invoke({
        "input": prompt1,
        "context": similar_docs
    })
    
    #retriever = st.session_state.vectors.as_retriever()
    #retrieval_chain = create_retrieval_chain(retriever, document_chain)
    #response = retrieval_chain.invoke({"input": prompt1})
    
     # Display the response
    st.write(response)
    
    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"Execution time: {execution_time:.2f} seconds")   
    
    
    #with st.expander("Document Similarity Search"):
        #for i, doc in enumerate(response["context"]):
            #st.write(doc.page_content)
            #st.write("--------------------------------")
            
    with st.expander("Document Similarity Search"):
        for doc in similar_docs:
            st.write(doc.page_content)
            st.write("--------------------------------")