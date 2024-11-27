import streamlit as st
import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
import numpy as np

# Configure page settings
st.set_page_config(page_title="Document Q&A System", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.collection = None
    st.session_state.ef = None

def initialize_system():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ef = BGEM3EmbeddingFunction(use_fp16=False, device=device)
    
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

    try:
        collection_name = "hybrid_demo"
        col = Collection(name=collection_name)
        col.load()
        return ef, col
    except Exception as e:
        st.error(f"Failed to load collection: {e}")
        return None, None

def convert_sparse_vector(sparse_matrix):
    """Convert sparse matrix to Milvus sparse vector format"""
    # Get the indices and data from the sparse matrix
    coo = sparse_matrix.tocoo()
    indices = list(zip(coo.row, coo.col))
    values = coo.data.tolist()
    
    # Format for Milvus sparse vector
    return {
        "indices": indices,
        "values": values
    }

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
    
    # Convert sparse embedding to Milvus format
    sparse_vector = convert_sparse_vector(query_sparse_embedding)
    
    res = col.search(
        [sparse_vector],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def hybrid_search(col, query_dense_embedding, query_sparse_embedding, 
                 sparse_weight=1.0, dense_weight=1.0, limit=5):
    dense_search_params = {"metric_type": "IP", "params": {}}
    sparse_search_params = {"metric_type": "IP", "params": {}}
    
    # Convert sparse embedding to Milvus format
    sparse_vector = convert_sparse_vector(query_sparse_embedding)
    
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_req = AnnSearchRequest(
        [sparse_vector], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

def format_result(text, max_length=200):
    """Format the result text for display"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def main():
    st.title("Document Q&A System")
    
    # Initialize system if not already done
    if not st.session_state.initialized:
        st.session_state.ef, st.session_state.collection = initialize_system()
        if st.session_state.ef and st.session_state.collection:
            st.session_state.initialized = True
            st.success(f"Connected to collection with {st.session_state.collection.num_entities} documents")
    
    # Query interface
    st.subheader("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Searching for answers..."):
            try:
                # Generate embeddings for query
                query_embeddings = st.session_state.ef([query])
                
                # Get dense vector
                dense_vector = query_embeddings["dense"][0].tolist()  # Convert to list if needed
                
                # Get sparse vector - using the first row of the sparse matrix
                sparse_vector = query_embeddings["sparse"][0]  # This is a sparse matrix
                
                # Perform searches
                with st.spinner("Performing dense search..."):
                    dense_results = dense_search(st.session_state.collection, dense_vector)
                
                with st.spinner("Performing sparse search..."):
                    sparse_results = sparse_search(st.session_state.collection, sparse_vector)
                
                with st.spinner("Performing hybrid search..."):
                    hybrid_results = hybrid_search(
                        st.session_state.collection,
                        dense_vector,
                        sparse_vector,
                        sparse_weight=0.7,
                        dense_weight=1.0,
                    )
                
                # Display results
                st.subheader("Search Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("Dense Search Results:")
                    for idx, result in enumerate(dense_results, 1):
                        st.markdown(f"**Result {idx}:**")
                        st.write(format_result(result))
                        st.markdown("---")
                
                with col2:
                    st.write("Sparse Search Results:")
                    for idx, result in enumerate(sparse_results, 1):
                        st.markdown(f"**Result {idx}:**")
                        st.write(format_result(result))
                        st.markdown("---")
                
                with col3:
                    st.write("Hybrid Search Results:")
                    for idx, result in enumerate(hybrid_results, 1):
                        st.markdown(f"**Result {idx}:**")
                        st.write(format_result(result))
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")
                st.error("Please try again with a different query.")

if __name__ == "__main__":
    main()
