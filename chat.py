import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the Groq and Google Generative AI API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")


def initialize_llm():
    """Initialize the language model"""
    try:
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma-7b-it"
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def initialize_pinecone():
    """Initialize Pinecone client"""
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        return pc
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None
   

def get_similar_docs(query, index, embeddings, k=5):
    """Retrieve similar documents from Pinecone"""
    query_embedding = embeddings.embed_query(query)
    
    results = index.query(
        namespace='',
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    print("============== results ==========================")
    print(results)
    print("================ results ========================")
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
    print("================ documents ========================")
    print(documents)
    print("================== documents ======================")
    return documents

def generate_response(user_input, llm, index, embeddings):
    """Generate response using LLM and context from Pinecone"""
    similar_docs = get_similar_docs(user_input, index, embeddings)
    
    context = "\n".join([doc.page_content for doc in similar_docs])
    
    print(context)
     
    prompt = ChatPromptTemplate.from_template(
    """System: You are a legal research assistant specializing in finding accurate information from legal documents.

    Context: {context}

    User Question: {question}

    Instructions:
    1. Carefully analyze the provided context to ensure a precise understanding.
    2. Only use the facts and information presented in the context to respond.
    3. Provide a direct, clear, and concise answer to the user's question.
    4. If the context lacks enough information, explicitly mention the insufficiency and suggest a way to acquire more details.
    5. Where applicable, highlight the relevant law, regulation, or legal precedent that supports your answer.

    Response:
    """)
    
    print(prompt)

    response = llm.invoke(prompt.format(
        context=context,
        question=user_input
    ))
    print(response)
    return response.content, similar_docs

def main():
    st.title("Chat with Real AI")
    st.write("Ask me anything about DCPR 2034")
    
    # Initialize components
    llm = initialize_llm()
    if llm is None:
            st.error("Failed to initialize the language model. Please check your API key and try again.")
            return
        
    pc = initialize_pinecone()
    index = pc.Index("realincgemma")  # Replace with your index name
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    # Chat input
    user_input = st.text_input("Type your question here...")
    
    if user_input:
        # Generate and display response
        start_time = time.time()
        with st.spinner("Generating answer..."):
            response, similar_docs = generate_response(user_input, llm, index, embeddings)
            st.write("Answer:")
            st.write(response)
            
            end_time = time.time()
            execution_time = end_time - start_time
            st.write(f"Response time: {execution_time:.2f} seconds")
        
        # Optional: Show relevant documents
        with st.expander("View Related Documents"):
            for i, doc in enumerate(similar_docs, 1):
                st.write(f"Document {i}:")
                st.write(doc.page_content)
                st.write("---")

if __name__ == "__main__":
    main()
