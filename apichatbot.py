import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv

import requests
from pydantic import BaseModel
from typing import List
import threading
from flask import Flask, request, jsonify


# Load environment variables
load_dotenv()

# Load the Groq and Google Generative AI API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Flask app
app = Flask(__name__)
    
def initialize_llm():
    """Initialize the language model"""
    try:
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it"
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
   

def get_similar_docs(query, index, embeddings, k=3):
    """Retrieve similar documents from Pinecone"""
    query_embedding = embeddings.embed_query(query)
    
    results = index.query(
        namespace='',
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    #print("============== results ==========================")
    #print(results)
    #print("================ results ========================")
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
    #print("================ documents ========================")
    #print(documents)
    #print("================== documents ======================")
    return documents

def generate_response(user_input, llm, index, embeddings):
    """Generate response using LLM and context from Pinecone"""
    try:
        similar_docs = get_similar_docs(user_input, index, embeddings)
    
        context = "\n".join([doc.page_content for doc in similar_docs])
    
        #print(context)
     
        prompt = ChatPromptTemplate.from_template(
        """System: You are a helpful Expert Legal AI assistant focused on providing accurate and relevant information related to legal documents.

        Context: {context}

        User Question: {question}

        Instructions:
        1. Analyze the context carefully, keeping in mind the legal and regulatory nature of the document.
        2. Consider only facts presented in the context without introducing assumptions or external information.
        3. Provide a clear, concise, and direct answer aligned with legal terminology and structure.
        4. If information is insufficient to answer the query, state so explicitly and suggest reviewing the full document for more details.

        Response:
        """)

        response = llm.invoke(prompt.format(
            context=context,
            question=user_input
        ))
        #print(response)
        return response.content, similar_docs
    except Exception as e:
        
        return "I apologize, but I encountered an error generating the response.", []



# Define request and response models
class QueryRequest(BaseModel):
    query: str

class DocumentInfo(BaseModel):
    source: str
    page: int
    content: str

class QueryResponse(BaseModel):
    answer: str
    relevant_documents: List[DocumentInfo]


@app.route('/api/query', methods=['POST'])
def query_endpoint():
    """API endpoint to handle queries"""
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data or 'query' not in request_data:
            return jsonify({'error': 'No query provided'}), 400

        query = request_data['query']
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Initialize components
        llm = initialize_llm()
        if llm is None:
            return jsonify({'error': 'Failed to initialize language model'}), 500

        pc = initialize_pinecone()
        if pc is None:
            return jsonify({'error': 'Failed to initialize Pinecone'}), 500

        index = pc.Index("realincgemma")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

        # Generate response
        response, similar_docs = generate_response(query, llm, index, embeddings)

        # Format response
        relevant_docs = [{
            'source': doc.metadata['source'],
            'page': doc.metadata['page'],
            'content': doc.page_content
        } for doc in similar_docs]

        return jsonify({
            'answer': response,
            'relevant_documents': relevant_docs
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def run_flask(host, port):
    """Run the Flask server"""
    app.run(host=host, port=port)
     
    
    
def main():
    
    """Main function with option to run web UI or API server"""
    #import argparse
    
    #parser = argparse.ArgumentParser(description="Run the chatbot as web UI or API server")
    #parser.add_argument("--mode", choices=["web", "api"], default="web",
                       #help="Run mode: 'web' for Streamlit UI or 'api' for REST API")
    
    
    #args = parser.parse_args()
    
    #if args.mode == "web":
    st.title("Chat with Real AI")
    st.write("Ask me anything about DCPR 2034")
    
     # Get the port from environment variable or use default
    port = 10000
    host = os.getenv("HOST", "0.0.0.0")
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(
        target=run_flask,
        args=(host, port),
        daemon=True
    )
    flask_thread.start()
    
    
     # Add a sidebar to show API information
    with st.sidebar:
        st.header("API Information")
        api_url = f"http://0.0.0.0:10000/api/query"
        st.write(f"API Endpoint: {api_url}")
        
         # Add API test form
        api_test_query = st.text_input("Test API Query")
        if st.button("Test API"):
        
                response = requests.post(
                    api_url,
                    json={"query": api_test_query}
                )
                if response.status_code == 200:
                    st.json(response.json())
                else:
                    st.error(f"API Error: {response.status_code}")
          
                
    # Chat input and history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "user_input" not in st.session_state:
        st.session_state.input_value = ""
        
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    # User input
    user_input = st.chat_input(
        "Type your message here...", key="chat_input"
    )           
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
    
            
    # Create a chat input
    
    if user_input:  
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display spinner while generating response
        with st.spinner("Generating Answer...."):
            response, similar_docs = generate_response(user_input, llm, index, embeddings)
        
        with st.chat_message("assistant"):
            st.markdown(response)    
            st.session_state.messages.append({"role": "assistant", "content": response})
                
                        
        # Clear the input field
        st.session_state.user_input = ""
    #else:
        #run_api_server()
        
    
    

if __name__ == "__main__":
    main()
