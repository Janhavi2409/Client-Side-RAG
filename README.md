# Client-Side RAG with React, Ollama and Semantic Search

This is a proof-of-concept application that demonstrates how to implement retrieval-augmented generation (RAG) entirely on the client side using:

1. React for the frontend UI
2. Ollama for local LLM inference
3. Semantic search using embeddings for efficient document retrieval

## Key Features

- **Document Upload**: Upload text or CSV files to build your knowledge base
- **Semantic Search**: Find relevant documents based on meaning, not just keywords
- **Local LLM Integration**: Generate answers based on retrieved context using Ollama
- **Completely Client-Side**: No server required - all processing happens in the browser

## Prerequisites

- [Ollama](https://ollama.ai/) must be installed and running on your machine
- You'll need at least one LLM model pulled in Ollama (llama3, mistral, phi3, or gemma)
- The nomic-embed-text model for embeddings will be automatically pulled if needed

## How It Works

1. **Document Processing**:
   - Upload documents (text or CSV)
   - Each document is embedded using an embeddings model via Ollama
   - Embeddings capture the semantic meaning of each document

2. **Semantic Search**:
   - User queries are also embedded
   - Cosine similarity is calculated between query and document embeddings
   - Most similar documents are retrieved as context

3. **Answer Generation**:
   - Retrieved documents are used as context for the LLM
   - The LLM generates an answer based only on the provided context
   - This ensures responses are grounded in your knowledge base

## Implementation Details

- Uses the Ollama API to communicate with local models
- Implements cosine similarity for vector comparison
- Supports both semantic search and fallback keyword search
- Efficiently processes documents and generates embeddings

## Getting Started

1. Install and start Ollama
2. Make sure you have pulled at least one of the supported models (llama3, mistral, phi3, gemma)
3. Run this React application
4. Upload your documents
5. Search and ask questions!

## Technical Considerations

- Large document collections may require batching or pagination
- Embeddings are stored in memory - a production version would need persistence
- This PoC demonstrates the concept but may need optimization for large datasets