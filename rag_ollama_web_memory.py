import requests
from bs4 import BeautifulSoup
import os
from langchain_community.document_loaders import WebBaseLoader

# Import Ollama specific components
from langchain_community.embeddings import OllamaEmbeddings
# IMPORTANT: Use ChatOllama for conversational capabilities
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import for Conversational Memory and Chains
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough


# --- Configuration ---
# List of URLs to scrape. Replace with your desired websites.
SOURCE_URLS = [
    "https://en.wikipedia.org/wiki/Multimedia_University",
    "https://www.mmu.edu.my/leadership",
    # Add more URLs here
]

# Ollama models
OLLAMA_LLM = "llama3.1" # Or another LLM you have pulled
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large" # Or another embedding model you have pulled

# Directory for persistent vector store
CHROMA_DB_DIR = "./chroma_db_ollama_memory" # Use a different directory name

# In-memory storage for chat history (per session)
store = {} # Dictionary to hold chat histories keyed by session ID

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Returns the chat message history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Main RAG Implementation ---

def main():
    """
    Main function to perform RAG with conversational memory using Ollama.
    """
    # --- 1. Load and Process Documents ---
    # (This part remains the same as before)
    print("Loading documents from URLs...")
    all_documents = []
    for url in SOURCE_URLS:
        try:
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    split_documents = text_splitter.split_documents(all_documents)
    print(f"Split into {len(split_documents)} chunks.")

    # --- 2. Create/Load Vector Store ---
    print("Initializing Ollama Embeddings...")
    try:
        embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        # Test embedding creation
        _ = embedding_function.embed_query("test embedding")
        print("Ollama Embeddings initialized successfully.")
    except Exception as e:
         print(f"Error initializing Ollama Embeddings. Make sure Ollama is running and model '{OLLAMA_EMBEDDING_MODEL}' is pulled.")
         print(e)
         return

    if os.path.exists(CHROMA_DB_DIR):
        print(f"Loading existing Chroma DB from {CHROMA_DB_DIR}")
        try:
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
            print("Chroma DB loaded successfully.")
        except Exception as e:
            print(f"Error loading Chroma DB: {e}. Rebuilding.")
            import shutil
            shutil.rmtree(CHROMA_DB_DIR)
            vectorstore = Chroma.from_documents(documents=split_documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)
            print("Chroma DB rebuilt.")
    else:
        print(f"Creating new Chroma DB at {CHROMA_DB_DIR}")
        vectorstore = Chroma.from_documents(documents=split_documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)
        print("Chroma DB created.")

    # --- 3. Setup RAG Chain with Memory ---
    print(f"Initializing Ollama LLM using model: {OLLAMA_LLM}")
    try:
        # Use ChatOllama for conversational capabilities with history
        ollama_llm = ChatOllama(
            model=OLLAMA_LLM,
            temperature=0.1 # Keep temperature low for factual answers
            # You might need to add base_url if Ollama is not on default host/port
            # base_url="http://localhost:11434"
        )
        # Test LLM call (optional but good for debugging)
        # response = ollama_llm.invoke("Hi")
        # print(f"Ollama LLM test response: {response.content[:50]}...")
        print("Ollama LLM initialized successfully.")
        print(f"Ensure Ollama server is running and model '{OLLAMA_LLM}' is pulled.")

    except Exception as e:
        print(f"Error initializing Ollama LLM. Make sure Ollama server is running")
        print(f"and model '{OLLAMA_LLM}' is pulled.")
        print(e)
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Fetch top 5 relevant chunks

    # Prompt for the history-aware retriever (rewrites the question)
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a standalone question for search. Return only the question.") # Clarified instruction
    ])

    # Create the history-aware retriever chain
    history_aware_retriever_chain = create_history_aware_retriever(
        ollama_llm,      # Use the Ollama LLM
        retriever,
        history_aware_retriever_prompt
    )

    # Prompt for the final RAG answer generation
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful chatbot to answer MMU related questions.
         Answer the user's question based ONLY on the below context and the chat history.
         If you cannot find the answer in the provided context or chat history, state that you do not have enough information from the provided sources to answer the question. 
         Do not make up an answer. 
         If the context is not relevant to the question, state that the context is not relevant.
         Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # Create the document combination chain
    stuff_documents_chain = create_stuff_documents_chain(
        ollama_llm, # Use the Ollama LLM
        rag_prompt
    )

    # Create the overall RAG chain
    conversational_rag_chain = create_retrieval_chain(
        history_aware_retriever_chain,
        stuff_documents_chain
    )

    # Wrap the RAG chain with RunnableWithMessageHistory to automatically manage chat history
    with_message_history = RunnableWithMessageHistory(
        conversational_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("\nConversational RAG system (Ollama) is ready. Ask me a question about the scraped content.")
    print("Type 'quit' to exit.")
    print("Try asking follow-up questions!")

    # --- 4. Querying Loop ---
    session_id = "abc" # Fixed session ID for this script

    while True:
        question = input("\nYour Question: ")
        if question.lower() == 'quit':
            break

        if not question.strip():
            continue

        try:
            response = with_message_history.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )

            print("\nAnswer:")
            print(response['answer'])

        except Exception as e:
            print(f"An error occurred during the RAG process: {e}")
            print("Please ensure Ollama server is running with the correct models pulled.")
            print(e)


    print("Exiting.")

if __name__ == "__main__":
    main()