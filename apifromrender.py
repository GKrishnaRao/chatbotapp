import os
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
import threading

# Your existing imports...

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class DocumentInfo(BaseModel):
    source: str
    page: int
    content: str

class QueryResponse(BaseModel):
    answer: str
    relevant_documents: List[DocumentInfo]

async def process_query(query: str):
    """Process the query and return response with relevant documents"""
    try:
        llm = initialize_llm()
        if llm is None:
            raise HTTPException(status_code=500, detail="Failed to initialize language model")
        
        pc = initialize_pinecone()
        index = pc.Index("realincgemma")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        response, similar_docs = generate_response(query, llm, index, embeddings)

        relevant_docs = [
            DocumentInfo(
                source=doc.metadata['source'],
                page=doc.metadata['page'],
                content=doc.page_content
            ) for doc in similar_docs
        ]

        return QueryResponse(
            answer=response,
            relevant_documents=relevant_docs
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """API endpoint to handle queries"""
    return await process_query(request.query)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_api(port):
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=port)

def main():
    st.title("Chat with Real AI")
    st.write("Ask me anything about DCPR 2034")

    # Get the port from environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=lambda: run_api(port), daemon=True)
    api_thread.start()

    # Display API Information
    with st.sidebar:
        st.header("API Information")
        api_url = f"https://your-render-app-name.onrender.com/api/query"
        st.write(f"API Endpoint: {api_url}")
        
        st.markdown("""
        ### API Documentation
        
        **Endpoint:** `/api/query`
        
        **Method:** POST
        
        **Request Body:**
        ```json
        {
            "query": "Your question here"
        }
        ```
        
        **Response:**
        ```json
        {
            "answer": "AI generated response",
            "relevant_documents": [
                {
                    "source": "document source",
                    "page": 1,
                    "content": "relevant content"
                }
            ]
        }
        ```
        """)

    # Your existing Streamlit UI code...

if __name__ == "__main__":
    main()
