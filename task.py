import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber

# 1. Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = [page.extract_text() for page in pdf.pages]
    return text

# 2. Chunk Text for Granularity
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"Chunks: {chunks}")  # Debugging: print the chunks
    return chunks

# 3. Embed Chunks Using Sentence-Transformers
def embed_text(chunks, model):
    if not chunks:
        print("No chunks to embed!")  # Debugging: print if no chunks are available
    embeddings = model.encode(chunks, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")  # Debugging: print embeddings shape
    return embeddings

# 4. Create FAISS Index
def create_faiss_index(embeddings):
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings to add to FAISS index.")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Using L2 distance
    index.add(embeddings)
    print(f"FAISS index created with dimension: {dim}")
    return index

# 5. Retrieve Relevant Chunks from FAISS Index
def retrieve_relevant_chunks(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    print(f"Query embedding shape: {query_embedding.shape}")  # Debugging: print query embedding shape

    # Check if the FAISS index and query embedding have matching dimensions
    index_dim = index.d
    if query_embedding.shape[1] != index_dim:
        raise ValueError(f"Dimension mismatch: query embedding has dimension {query_embedding.shape[1]} but FAISS index has dimension {index_dim}.")

    distances, indices = index.search(query_embedding, top_k)
    print(f"Distances: {distances}, Indices: {indices}")  # Debugging: print distances and indices
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# 6. Generate Response (Placeholder Function)
def generate_response(relevant_chunks, query):
    # For simplicity, return a concatenation of relevant chunks
    response = "\n".join(relevant_chunks)
    return response

# Main Pipeline for Full PDF
def rag_pipeline(pdf_path, query, chunk_size=300, top_k=5):
    # Step 1: Extract Text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    full_text = " ".join(pdf_text)

    # Step 2: Chunk Text
    chunks = chunk_text(full_text, chunk_size)
    if not chunks:
        print("No text chunks found!")  # Debugging: print if no chunks are found

    # Step 3: Embed Chunks
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_text(chunks, model)

    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings generated. Please check the text extraction and chunking process.")

    # Step 4: Create FAISS Index
    index = create_faiss_index(embeddings)
    print(f"FAISS index dimension: {index.d}")

    # Step 5: Retrieve Relevant Chunks
    relevant_chunks = retrieve_relevant_chunks(query, model, index, chunks, top_k)

    # Step 6: Generate Response
    response = generate_response(relevant_chunks, query)
    return response

# Example Usage for Specific Tasks
if __name__ == "__main__":
    # Path to the example PDF
    pdf_path = r"C:\Users\Devarshini\tables-charts-and-graphs-with-examples-from.pdf"

    # Specific queries
    queries = [
        "Extract unemployment information based on the type of degree from page 2.",
        "Extract the tabular data from page 6."
    ]

    # Step 1: Extract text from specific pages
    pdf_text = extract_text_from_pdf(pdf_path)
    page_2_text = pdf_text[1]  # Page index starts at 0
    page_6_text = pdf_text[5]

    # Step 2: Handle page 2 query (Unemployment info)
    print(f"Page 2 text: {page_2_text[:500]}")  # Debugging: print the first 500 characters of text from page 2
    unemployment_chunks = chunk_text(page_2_text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    unemployment_embeddings = embed_text(unemployment_chunks, model)

    if unemployment_embeddings.shape[0] == 0:
        print("No embeddings generated for unemployment data.")  # Debugging: print if no embeddings are generated for unemployment

    # Check if embeddings are generated before proceeding
    if unemployment_embeddings.shape[0] > 0:
        unemployment_index = create_faiss_index(unemployment_embeddings)
        print(f"Unemployment FAISS index dimension: {unemployment_index.d}")
        unemployment_relevant_chunks = retrieve_relevant_chunks(queries[0], model, unemployment_index, unemployment_chunks)
        unemployment_response = generate_response(unemployment_relevant_chunks, queries[0])
        print("\nUnemployment Information:\n", unemployment_response)
    else:
        print("Skipping FAISS index creation due to empty embeddings.")
    
    # Step 3: Handle page 6 query (Tabular data)
    with pdfplumber.open(pdf_path) as pdf:
        page_6 = pdf.pages[5]
        table_data = page_6.extract_table()  # Extracts the table as a list of rows
    print("\nTabular Data from Page 6:\n", table_data)
