import requests
from bs4 import BeautifulSoup
import os
from langchain_community.document_loaders import WebBaseLoader
# Import for loading local files
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Import Ollama specific components
from langchain_community.embeddings import OllamaEmbeddings
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

# Local Directory for Text Files
# IMPORTANT: Create this directory and place your .txt files inside
LOCAL_DOCS_DIR = "./local_docs"

# Ollama models
OLLAMA_LLM = "llama3.1" # Or another LLM you have pulled
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" # Or another embedding model you have pulled

# Directory for persistent vector store
CHROMA_DB_DIR = "./chroma_db_ollama_memory"

# In-memory storage for chat history (per session)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Returns the chat message history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Main RAG Implementation ---

def main():
    """
    Main function to perform RAG based on web and local text files using Ollama.
    """
    # --- 1. Load Documents ---
    print("Loading documents from URLs...")
    all_documents = []
    # Load from URLs
    for url in SOURCE_URLS:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            if docs:
                print(f"Loaded from web: {url}")
                all_documents.extend(docs)
            else:
                 print(f"Could not load documents from: {url}")

        except Exception as e:
            print(f"Error loading {url} using WebBaseLoader: {e}")

    # Load from local directory
    print(f"Loading documents from local directory: {LOCAL_DOCS_DIR}...")
    if os.path.exists(LOCAL_DOCS_DIR):
        try:
            # Use DirectoryLoader with TextLoader for .txt files
            loader = DirectoryLoader(LOCAL_DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
            local_docs = loader.load()
            if local_docs:
                print(f"Loaded {len(local_docs)} documents from {LOCAL_DOCS_DIR}")
                all_documents.extend(local_docs)
            else:
                 print(f"No .txt documents found in {LOCAL_DOCS_DIR}")
        except Exception as e:
            print(f"Error loading documents from local directory {LOCAL_DOCS_DIR}: {e}")
    else:
        print(f"Local documents directory not found: {LOCAL_DOCS_DIR}. Skipping local file loading.")


    if not all_documents:
        print("No documents were loaded successfully from any source. Exiting.")
        return

    print(f"Loaded {len(all_documents)} documents in total from all sources.")

    # --- 2. Process Documents (Splitting, Embedding, Vector Store) ---
    # (This part processes the combined list of documents and remains the same)
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)
    print(f"Split into {len(split_documents)} chunks.")

    print("Initializing Ollama Embeddings...")
    try:
        embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
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
    # (This part remains the same, using the vectorstore that now includes file data)
    print(f"Initializing Ollama LLM using model: {OLLAMA_LLM}")
    try:
        ollama_llm = ChatOllama(
            model=OLLAMA_LLM,
            temperature=0.1
            # base_url="http://localhost:11434" # Uncomment if needed
        )
        print("Ollama LLM initialized successfully.")
        print(f"Ensure Ollama server is running and model '{OLLAMA_LLM}' is pulled.")

    except Exception as e:
        print(f"Error initializing Ollama LLM. Make sure Ollama server is running")
        print(f"and model '{OLLAMA_LLM}' is pulled.")
        print(e)
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a standalone question for search. Return only the question.")
    ])

    history_aware_retriever_chain = create_history_aware_retriever(
        ollama_llm,
        retriever,
        history_aware_retriever_prompt
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the user's question based ONLY on the below context and the chat history.
If you cannot find the answer in the provided context or chat history, state that you do not have enough information from the provided sources to answer the question. Do not make up an answer.

Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(
        ollama_llm,
        rag_prompt
    )

    conversational_rag_chain = create_retrieval_chain(
        history_aware_retriever_chain,
        stuff_documents_chain
    )

    with_message_history = RunnableWithMessageHistory(
        conversational_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer", # Added in previous fix
    )

    print("\nConversational RAG system (Ollama + Local Files) is ready. Ask me a question.")
    print("Type 'quit' to exit.")
    print("Sources include scraped websites and local .txt files.")

    # --- 4. Querying Loop ---
    session_id = "abc"

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

            # Optional: Print sources (requires modifying the rag_prompt or chain)
            # Currently, the retrieved 'context' is not directly in the final response dict
            # You could modify the chain to return context explicitly if needed.
            # print("\nSources (based on chunks used):")
            # for doc in response.get('context', []): # If context were returned
            #     print(f"- {doc.metadata.get('source', 'Unknown Source')}")


        except Exception as e:
            print(f"An error occurred during the RAG process: {e}")
            print("Please ensure Ollama server is running with the correct models pulled.")
            print(e)

    print("Exiting.")

if __name__ == "__main__":
    main()