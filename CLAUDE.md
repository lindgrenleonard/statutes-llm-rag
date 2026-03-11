# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG chatbot for the IT Chapter's statutes and organizational memos. Uses LlamaIndex with a local Ollama LLM to answer questions about governance documents stored as markdown in `documents/`.

## Architecture

`app.py` is the entire application — a single-file pipeline:
1. Loads 28 markdown documents from `documents/` via `SimpleDirectoryReader`
2. Splits documents using `SemanticSplitterNodeParser` (embedding-based topic boundary detection)
3. Builds an in-memory `VectorStoreIndex` using `BAAI/bge-m3` embeddings (HuggingFace)
4. Retrieves top 20 candidates, then reranks to top 5 using `cross-encoder/ms-marco-MiniLM-L-6-v2`
5. Creates a chat engine backed by a local Ollama model (`qwen3.5:4b`)
6. Runs an interactive chat REPL

The `Modelfile` defines a custom Ollama model with system prompt and tuned parameters (temperature 0.3).

## Running

```bash
# Requires Ollama running locally:
ollama pull qwen3.5:4b
ollama serve  # if not already running

# Install dependencies:
pip install -r requirements.txt

python app.py
```

No test suite, linter, or build system is configured.

## Key Details

- Documents are in English (only the filenames are in Swedish)
- Retrieval pipeline: vector search (top 20) → cross-encoder rerank (top 5) → LLM response
