# Local RAG Assistant with Microsoft Foundry Local

## Overview

This project is a fully local and offline Retrieval-Augmented Generation (RAG) application built with Microsoft Foundry Local.

The system reads PDF documents, splits them into chunks, generates embeddings, stores them in SQLite, retrieves the most relevant chunks, and generates grounded answers using a local language model.

## Architecture

PDF Documents
-> Text Extraction
-> Chunking
-> Embedding Generation
-> SQLite Database
-> User Question
-> Query Embedding
-> Cosine Similarity Search
-> Top-3 Relevant Chunks
-> Qwen2.5-0.5B (Fast) or Qwen3-4B (Quality)
-> Answer + Source Citation

## Models

- Embedding model: qwen3-embedding-0.6b
- Fast answer model: qwen2.5-0.5b
- Quality answer model: qwen3-4b
- Runtime: Microsoft Foundry Local

## Main Files

- multi_pdf_ingest.py - Processes PDFs and stores chunks and embeddings in SQLite.
- multi_pdf_rag.py - Main Local RAG application.
- web_app.py - Local-only web server and RAG API.
- web/ - Offline interface files. No CDN or internet connection is used.
- evaluate_retrieval.py - Tests retrieval performance.
- evaluate_rag.py - Runs end-to-end evaluation.
- evaluation_questions.json - Contains evaluation questions.
- evaluation_results.csv - Stores evaluation results.
- performance_test.py - Measures RAG execution times.

## Installation

Install Python dependencies with:

pip install -r requirements.txt

Download the required Foundry Local models:

foundry model download qwen3-embedding-0.6b
foundry model download qwen2.5-0.5b
foundry model download qwen3-4b

## Document Ingestion

Place PDF files in the documents folder.

Run:

python multi_pdf_ingest.py

## Running the Assistant

### Offline web interface

Double-click `Start_Local_RAG_Web.bat` in the `LocalRAG_Final` folder.

The application opens at:

http://localhost:8765

The server listens only on this computer. Interface assets, document retrieval,
embeddings, and answer generation all run locally. Keep the terminal window open
while using the site, then press `Ctrl+C` to stop it.

Use the model selector above the question box to switch between the faster
Qwen2.5 0.5B model and the higher-quality Qwen3 4B model.

You can also start it from the project folder with:

python web_app.py

### Command-line interface

Run:

python multi_pdf_rag.py

Type exit or quit to close the application.

## Grounding and Citations

The assistant answers using retrieved document content.

The system retrieves the Top-3 most relevant chunks using cosine similarity.

A minimum retrieval similarity score of 0.35 is used. If the best retrieved chunk has a score below this threshold, the question is rejected before answer generation and the system returns:

The information is not available in the provided sources.

This helps reduce unsupported answers for questions that are outside the local document collection.

When enough information is available, answers include source references such as:

[Source 1]

## Evaluation Results

- Retrieval Top-1 accuracy: 91.7%
- Retrieval Top-3 accuracy: 100%
- Answerable questions passed: 12/12
- Fallback questions passed: 3/3
- End-to-end evaluation: 15/15

These results apply only to the prepared evaluation dataset.

## Performance

The embedding model and answer model are loaded once when the application starts and remain in memory while the application is running.

This avoids loading and unloading the models for every question.

Example measurements on the test machine:

- Application startup: approximately 13.1 seconds
- Query embedding: approximately 2.6 seconds
- Retrieval search: approximately 0.012 seconds
- Qwen3-4B answer generation: approximately 45-51 seconds
- Total question time after startup: approximately 47-53 seconds

Retrieval itself is very fast. The main performance bottleneck is local Qwen3-4B answer generation using CPUExecutionProvider.

## Privacy

Document processing, retrieval, embedding generation, and language model inference are performed locally.

No cloud API is required for normal operation.
