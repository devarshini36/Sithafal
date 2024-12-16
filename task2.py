import openai
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Replace with your actual OpenAI API key
openai.api_key = 'your_openai_api_key'

# Step 1: Scrape content from websites
def scrape_website(url):
    print(f"Scraping {url}...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

# Step 2: Convert content to embeddings (using TF-IDF for simplicity)
def get_embeddings(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    embeddings = vectorizer.fit_transform(texts).toarray()
    return embeddings, vectorizer  # Return embeddings and the vectorizer for later query transformation

# Step 3: Retrieve relevant chunks
def retrieve_relevant_chunks(query, embeddings, texts, vectorizer, top_k=3):
    # Transform query into embedding
    query_embedding = vectorizer.transform([query]).toarray()[0]
    
    # Calculate cosine similarity between query and text embeddings
    similarities = []
    for text_embedding in embeddings:
        similarity = np.dot(query_embedding, text_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding))
        similarities.append(similarity)
    
    # Get top_k relevant texts
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [texts[i] for i in top_indices]
    return relevant_chunks

# Step 4: Generate response using OpenAI's GPT model
def generate_response(user_query, relevant_chunks):
    try:
        # Call OpenAI API for response generation using ChatCompletion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use gpt-4 or another model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": ' '.join(relevant_chunks)}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Step 5: Run the RAG pipeline
def run_rag_pipeline():
    # List of website URLs to scrape
    urls = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]
    
    # Scrape content from the websites
    website_texts = []
    for url in urls:
        website_texts.append(scrape_website(url))
    
    # Convert scraped content into embeddings
    embeddings, vectorizer = get_embeddings(website_texts)
    
    # Prompt user for a query
    user_query = input("Enter your query: ")

    # Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(user_query, embeddings, website_texts, vectorizer)

    # Generate a response based on the query and the relevant chunks
    response = generate_response(user_query, relevant_chunks)

    # Print the generated response
    if response:
        print("Response:", response)
    else:
        print("Sorry, I couldn't generate a response.")

if __name__ == "__main__":
    run_rag_pipeline()
