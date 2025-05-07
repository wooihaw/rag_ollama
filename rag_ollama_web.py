import requests
from bs4 import BeautifulSoup
import os
from langchain_community.document_loaders import WebBaseLoader # Alternative loader, easier
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Configuration ---
# List of URLs to scrape. Replace with your desired websites.
SOURCE_URLS = [
    "https://en.wikipedia.org/wiki/Multimedia_University",
    "https://www.mmu.edu.my/leadership",
    "https://www.mmu.edu.my/full-list-of-programmes-offered/"
    # Add more URLs here
]

# Ollama models
OLLAMA_LLM = "llama3.1" # Or another LLM you have pulled
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large" # Or another embedding model you have pulled

# Directory for persistent vector store
CHROMA_DB_DIR = "./chroma_db"

# --- Web Scraping Function (Basic) ---
# Using a simpler approach with LangChain's WebBaseLoader is often better,
# but keeping this simple function for demonstration if needed.
# def scrape_website(url):
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
#         soup = BeautifulSoup(response.content, 'html.parser')

#         # Extract text from relevant parts, e.g., body or main content area
#         # This is a basic extraction, you might need to customize this
#         text = soup.get_text()

#         # Clean up whitespace
#         text = ' '.join(text.split())
#         print(f"Successfully scraped: {url}")
#         return text
#     except requests.exceptions.RequestException as e:
#         print(f"Error scraping {url}: {e}")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred while scraping {url}: {e}")
#         return None

# --- Main RAG Implementation ---

def main():
    """
    Main function to perform RAG based on scraped websites.
    """
    # --- 1. Load and Process Documents ---
    print("Loading documents from URLs...")
    all_documents = []
    for url in SOURCE_URLS:
        try:
            # Use LangChain's WebBaseLoader for easier loading and basic parsing
            loader = WebBaseLoader(url)
            docs = loader.load()
            if docs:
                print(f"Loaded: {url}")
                all_documents.extend(docs)
            else:
                 print(f"Could not load documents from: {url}")

        except Exception as e:
            print(f"Error loading {url} using WebBaseLoader: {e}")


    if not all_documents:
        print("No documents were loaded successfully. Exiting.")
        return

    print(f"Loaded {len(all_documents)} documents in total.")

    print("Splitting documents into chunks...")
    # Split documents into smaller chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)
    print(f"Split into {len(split_documents)} chunks.")

    # --- 2. Create/Load Vector Store ---
    print("Initializing Ollama Embeddings...")
    try:
        ollama_embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        # Test embedding creation
        _ = ollama_embeddings.embed_query("test embedding")
        print("Ollama Embeddings initialized successfully.")
    except Exception as e:
         print(f"Error initializing Ollama Embeddings. Make sure Ollama is running and model '{OLLAMA_EMBEDDING_MODEL}' is pulled.")
         print(e)
         return


    # Check if vector store exists
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Loading existing Chroma DB from {CHROMA_DB_DIR}")
        try:
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=ollama_embeddings)
            print("Chroma DB loaded successfully.")
        except Exception as e:
            print(f"Error loading Chroma DB: {e}. Rebuilding.")
            # If loading fails, delete and rebuild
            import shutil
            shutil.rmtree(CHROMA_DB_DIR)
            vectorstore = Chroma.from_documents(documents=split_documents, embedding=ollama_embeddings, persist_directory=CHROMA_DB_DIR)
            print("Chroma DB rebuilt.")
    else:
        print(f"Creating new Chroma DB at {CHROMA_DB_DIR}")
        # Create the vector store from the split documents and embeddings
        vectorstore = Chroma.from_documents(documents=split_documents, embedding=ollama_embeddings, persist_directory=CHROMA_DB_DIR)
        print("Chroma DB created.")

    # --- 3. Setup RAG Chain ---
    print("Initializing Ollama LLM...")
    try:
        ollama_llm = Ollama(model=OLLAMA_LLM, temperature=0.1)
        # Test LLM call
        _ = ollama_llm.invoke("Hi")
        print("Ollama LLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ollama LLM. Make sure Ollama is running and model '{OLLAMA_LLM}' is pulled.")
        print(e)
        return

    # Define the retriever - how many relevant documents to fetch
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Fetch top 5 relevant chunks

    # Define the prompt template for the LLM
    prompt_template = ChatPromptTemplate.from_template("""
    You are an AI assistant that answers questions based ONLY on the provided context.
    If you cannot find the answer in the context, state that you don't have enough information.

    Context: {context}

    Question: {input}
    """)

    # Create a chain to combine documents and generate response
    document_chain = create_stuff_documents_chain(ollama_llm, prompt_template)

    # Create the retrieval chain
    # This chain first retrieves relevant documents and then passes them to the document_chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\nRAG system is ready. Ask me a question about the scraped content.")
    print("Type 'quit' to exit.")

    # --- 4. Querying Loop ---
    while True:
        question = input("\nYour Question: ")
        if question.lower() == 'quit':
            break

        if not question.strip():
            continue

        try:
            # Invoke the retrieval chain with the user's question
            # The chain handles retrieval and LLM call
            response = retrieval_chain.invoke({"input": question})

            # The response structure from create_retrieval_chain is typically {'input': ..., 'context': [...], 'answer': ...}
            print("\nAnswer:")
            print(response['answer'])

        except Exception as e:
            print(f"An error occurred during the RAG process: {e}")
            print("Please ensure Ollama is running and models are available.")


    print("Exiting.")

if __name__ == "__main__":
    main()