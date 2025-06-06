# Local RAG Project with Ollama and uv

This project demonstrates a Retrieval-Augmented Generation (RAG) system implemented locally using the Ollama framework to run an LLM (specifically Llama 3.1) and `uv` for fast dependency management and environment isolation.

## Table of Contents

-   [Features](#features)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
    -   [1. Install Ollama](#1-install-ollama)
    -   [2. Install uv](#2-install-uv)
-   [Setup](#setup)
    -   [1. Download the Llama 3.1 Model](#1-download-the-llama-31-model)
    -   [2. Project Dependencies](#2-project-dependencies)
-   [Running the Project](#running-the-project)
    -   [1. Clone the Repository](#1-clone-the-repository)
    -   [2. Install Project Dependencies](#2-install-project-dependencies)
    -   [3. Running the Python Scripts](#3-running-the-python-scripts)

## Features

* Local execution of RAG using Ollama.
* Leverages the Llama 3.1 Large Language Model.
* Uses `uv` for efficient and isolated dependency management.
* Provides a foundation for building local LLM applications.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Python 3.10+**: You can download Python from [python.org](https://www.python.org/downloads/). Ensure Python is added to your system's PATH.
* **Git**: Required to clone this repository. Download from [git-scm.com](https://git-scm.com/downloads).

## Installation

Follow these steps to get the project running on your local machine.

### 1. Install Ollama

Ollama is required to serve the Llama 3.1 model locally.

* **For Windows:**
    1.  Go to the official Ollama website: [ollama.com](https://ollama.com/).
    2.  Click on "Download" and select the Windows installer.
    3.  Run the installer and follow the on-screen instructions. Ollama will automatically start and run in the background.

* **For Linux:**
    1.  Open your terminal.
    2.  Run the following command to download and install Ollama:
        ```bash
        curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
        ```
    3.  This script will add Ollama to your system and start it as a service.

Verify the installation by opening a new terminal or command prompt and running:

```bash
ollama --version
```

### 2. Install uv

`uv` is a fast Python package installer and environment manager. 

* **For Windows:**
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```

* **For Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

Verify the installation:

```bash
uv --version
```

## Setup

Once Ollama and `uv` are installed, you need to set up the LLM and project dependencies.

### 1. Download the Llama 3.1 Model

The project requires both the LLM (`llama3.1`) and an embedding model (`nomic-embed-text`). Download them using the Ollama command-line interface.

Open your terminal or command prompt and run the following commands:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

These commands will download the necessary models. The download sizes are significant and may take some time depending on your internet connection.

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/wooihaw/rag-ollama.git
cd rag-ollama
```

### 2. Install Project Dependencies

Use uv to create a virtual environment (if it doesn't exist) and install the project dependencies with the following command:

```bash
uv sync
```

This command will:

- Automatically create a virtual environment in `.venv` if it doesn't exist
- Install all dependencies from `pyproject.toml` (and `uv.lock` if present)

### 3. Running the Python Scripts

Once the environment and dependencies have created and installed, enter the following command in a terminal:

```bash
uv run rag_ollama_web_txt_memory.py
```

To launch the web app version, enter the following command in a terminal:
```bash
uv run streamlit run rag_ollama_streamlit.py
```
