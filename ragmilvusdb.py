import streamlit as st
import torch
from langchain_community.document_loaders import PDFPlumberLoader
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)

# Configure page settings
st.set_page_config(page_title="Document Q&A System", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.collection = None
    st.session_state.ef = None

def initialize_system():
    # Set up device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Initialize BGE-M3 embedding function
    ef = BGEM3EmbeddingFunction(use_fp16=False, device=device)
    
    # Connect to Milvus
    try:
        connections.connect(
            alias="default",
            uri="https://in03-4e569f605c32eab.serverless.gcp-us-west1.cloud.zilliz.com",
            token="99cd003de770782d436a049c87fb669188dc4424443531a325043d7f42859ca8c3d058b952d2e92d33677cf72b4931d12150c29d"
        )
        st.success("Successfully connected to Milvus")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None, None

    # Create collection schema
    fields = [
        FieldSchema(
            name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"]),
    ]

    # Define collection
    collection_name = "hybrid_demo"
    schema = CollectionSchema(fields)

    # Create or get existing collection
    try:
        col = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            consistency_level="Strong"
        )
        
        # Create indices
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        col.create_index("dense_vector", dense_index)
        col.load()
        
        return ef, col
    except Exception as e:
        st.error(f"Failed to initialize collection: {e}")
        return None, None

def process_pdf(file, ef, col):
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    
    # Load PDF
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()
    doc_texts = [page.page_content for page in docs]
    
    # Generate embeddings
    docs_embeddings = ef(doc_texts)
    
    # Insert documents in batches
    for i in range(0, len(doc_texts), 50):
        batched_texts = doc_texts[i:i+50]
        batched_sparse_vectors = docs_embeddings["sparse"][i:i+50]
        batched_dense_vectors = docs_embeddings["dense"][i:i+50]
        
        batched_entities = [
            batched_texts,
            batched_sparse_vectors,
            batched_dense_vectors,
        ]
        
        col.insert(batched_entities)
    
    return len(doc_texts)

def dense_search(col, query_dense_embedding, limit=5):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def sparse_search(col, query_sparse_embedding, limit=5):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def hybrid_search(col, query_dense_embedding, query_sparse_embedding, 
                 sparse_weight=1.0, dense_weight=1.0, limit=5):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

def main():
    st.title("Document Q&A System")
    
    # Initialize system if not already done
    if not st.session_state.initialized:
        st.session_state.ef, st.session_state.collection = initialize_system()
        if st.session_state.ef and st.session_state.collection:
            st.session_state.initialized = True
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                num_pages = process_pdf(uploaded_file, st.session_state.ef, 
                                     st.session_state.collection)
                st.success(f"Processed {num_pages} pages successfully!")
    
    # Query interface
    st.subheader("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Searching for answers..."):
            # Generate embeddings for query
            query_embeddings = st.session_state.ef([query])
            
            # Perform searches
            dense_results = dense_search(st.session_state.collection, 
                                      query_embeddings["dense"][0])
            sparse_results = sparse_search(st.session_state.collection, 
                                        query_embeddings["sparse"][0])
            hybrid_results = hybrid_search(
                st.session_state.collection,
                query_embeddings["dense"][0],
                query_embeddings["sparse"][0],
                sparse_weight=0.7,
                dense_weight=1.0,
            )
            
            # Display results
            st.subheader("Search Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Dense Search Results:")
                for idx, result in enumerate(dense_results, 1):
                    st.write(f"{idx}. {result[:200]}...")
            
            with col2:
                st.write("Sparse Search Results:")
                for idx, result in enumerate(sparse_results, 1):
                    st.write(f"{idx}. {result[:200]}...")
            
            with col3:
                st.write("Hybrid Search Results:")
                for idx, result in enumerate(hybrid_results, 1):
                    st.write(f"{idx}. {result[:200]}...")

if __name__ == "__main__":
    main()
