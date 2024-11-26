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
    
    print(prompt)

    response = llm.invoke(prompt.format(
        context=context,
        question=user_input
    ))
    #print(response)
    return response.content, similar_docs

def main():
    st.title("Chat with Real AI")
    st.write("Ask me anything about DCPR 2034")
    
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

if __name__ == "__main__":
    main()
