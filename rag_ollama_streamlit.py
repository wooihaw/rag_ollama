import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import shutil

# --- Configuration ---
st.set_page_config(page_title="Conversational RAG with Ollama", layout="wide")
st.title("Conversational RAG with Ollama, Web & Local Docs 📚")

# List of URLs to scrape
SOURCE_URLS = [
    "https://en.wikipedia.org/wiki/Multimedia_University",
    "https://www.mmu.edu.my/leadership",
    # Add more URLs here
]

# Local Directory for Text Files
LOCAL_DOCS_DIR = "./local_docs"

# Ollama models
OLLAMA_LLM = "llama3.1"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Directory for persistent vector store
CHROMA_DB_DIR = "./chroma_db_ollama_memory_st"

# --- RAG Setup Function (Cached) ---
@st.cache_resource(show_spinner="Setting up the RAG chain... Please wait.")
def setup_rag_chain():
    """
    Sets up the entire RAG pipeline: loads documents, creates vector store,
    and initializes the conversational RAG chain.
    This function is cached to run only once per session.
    """
    all_documents = []

    # --- 1. Load Documents ---
    # Load from URLs
    with st.status("Loading documents from web URLs...", expanded=True) as status:
        for url in SOURCE_URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                if docs:
                    st.write(f"✓ Loaded from web: {url}")
                    all_documents.extend(docs)
                else:
                    st.write(f"✗ Could not load documents from: {url}")
            except Exception as e:
                st.error(f"Error loading {url}: {e}")
        status.update(label="Web loading complete!", state="complete")

    # Load from local directory
    with st.status(f"Loading documents from local directory: {LOCAL_DOCS_DIR}...", expanded=True) as status:
        if os.path.exists(LOCAL_DOCS_DIR):
            try:
                loader = DirectoryLoader(LOCAL_DOCS_DIR, glob="*.txt", loader_cls=TextLoader, show_progress=True)
                local_docs = loader.load()
                if local_docs:
                    st.write(f"✓ Loaded {len(local_docs)} documents from {LOCAL_DOCS_DIR}")
                    all_documents.extend(local_docs)
                else:
                    st.write(f"✗ No .txt documents found in {LOCAL_DOCS_DIR}")
            except Exception as e:
                st.error(f"Error loading documents from local directory: {e}")
        else:
            st.warning(f"Local documents directory not found: {LOCAL_DOCS_DIR}. Skipping.")
        status.update(label="Local loading complete!", state="complete")

    if not all_documents:
        st.error("No documents were loaded. The application cannot proceed. Please check your sources.")
        st.stop()

    # --- 2. Process Documents (Split, Embed, Store) ---
    with st.status("Splitting documents, embedding, and creating vector store...", expanded=True) as status:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(all_documents)
        st.write(f"Split into {len(split_documents)} chunks.")

        # Initialize Ollama Embeddings
        try:
            embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
            st.write("Ollama Embeddings initialized.")
        except Exception as e:
            st.error(f"Error initializing Ollama Embeddings. Make sure Ollama is running and model '{OLLAMA_EMBEDDING_MODEL}' is pulled.")
            st.error(e)
            st.stop()

        # Create or load Chroma DB
        if os.path.exists(CHROMA_DB_DIR):
            st.write(f"Loading existing Chroma DB from {CHROMA_DB_DIR}")
            try:
                vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
            except Exception as e:
                st.warning(f"Error loading Chroma DB: {e}. Rebuilding.")
                shutil.rmtree(CHROMA_DB_DIR)
                vectorstore = Chroma.from_documents(documents=split_documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)
        else:
            st.write(f"Creating new Chroma DB at {CHROMA_DB_DIR}")
            vectorstore = Chroma.from_documents(documents=split_documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)

        st.write("Vector store is ready.")
        status.update(label="Vector store created successfully!", state="complete")

    # --- 3. Setup RAG Chain with Memory ---
    with st.status("Initializing Ollama LLM and RAG chain...", expanded=True) as status:
        try:
            ollama_llm = ChatOllama(model=OLLAMA_LLM, temperature=0.1)
            st.write(f"Ollama LLM initialized with model: {OLLAMA_LLM}")
        except Exception as e:
            st.error(f"Error initializing Ollama LLM. Make sure Ollama server is running and model '{OLLAMA_LLM}' is pulled.")
            st.error(e)
            st.stop()

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Prompt for generating a standalone question
        history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Return only the query and nothing else."),
        ])

        history_aware_retriever_chain = create_history_aware_retriever(
            ollama_llm, retriever, history_aware_retriever_prompt
        )

        # Prompt for answering the question
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the user's question based ONLY on the below context.
If you cannot find the answer in the provided context, state that you do not have enough information from the provided sources to answer. Do not make up an answer.

Context:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        stuff_documents_chain = create_stuff_documents_chain(ollama_llm, rag_prompt)
        conversational_rag_chain = create_retrieval_chain(history_aware_retriever_chain, stuff_documents_chain)
        st.write("Conversational RAG chain is ready.")
        status.update(label="RAG chain setup complete!", state="complete")

    return conversational_rag_chain

# --- Main Application Logic ---

# Get the initialized RAG chain
try:
    rag_chain = setup_rag_chain()
except Exception as e:
    st.error("An error occurred during the setup of the RAG chain.")
    st.error(e)
    st.stop()


# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I am a RAG assistant. How can I help you today?")
    ]

# The main chat interface occupies the majority of the screen
# A container is used to group the chat messages
chat_container = st.container()

with chat_container:
    # Display existing chat messages
    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

# The chat input widget is anchored at the bottom
if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with chat_container:
         with st.chat_message("user"):
            st.markdown(prompt)

    # Process the user's question with the RAG chain
    with st.spinner("Thinking..."):
        try:
            # Prepare chat history for the chain
            chat_history_for_chain = [msg for msg in st.session_state.messages[:-1]]

            # Invoke the RAG chain
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": chat_history_for_chain
            })

            answer = response.get('answer', "Sorry, I could not generate an answer.")

            # Add AI response to session state and display it
            st.session_state.messages.append(AIMessage(content=answer))
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append(AIMessage(content=f"Sorry, I encountered an error: {e}"))
