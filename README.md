# Document QA System

This repository contains a Document QA (Question Answering) system that leverages OpenAI's GPT-3.5-turbo model and Chroma for embedding and vector storage. The system loads documents, splits them into chunks, generates embeddings, and stores them in a persistent vector database. It then allows querying the documents and generating responses using the OpenAI API.

## Features

- Load documents from a directory
- Split documents into chunks
- Generate embeddings using OpenAI's embedding functions
- Store embeddings in a persistent Chroma vector database
- Query documents and generate responses using OpenAI's GPT-3.5-turbo model

## Setup

### Prerequisites

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/installation/)
- [OpenAI API Key](https://beta.openai.com/signup/)
- [Chroma](https://www.trychroma.com/)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/amirkhier/document-openai-chromadb.git
    cd document-openai-chromadb
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a  file in the root directory and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Load and process documents:

    ```python
    python system.py
    ```

2. Follow the prompts to enter your questions and get responses.

## File Structure

- : Main script to load documents, generate embeddings, store them in Chroma, and query the documents.
- : Directory containing sample news articles to be processed.
- : List of required Python packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.