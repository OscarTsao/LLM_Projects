# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the project structure, conventions, and goals.

## Project Overview

This is a Python project that implements a Retrieval-Augmented Generation (RAG) system for matching social media posts against DSM-5 criteria.

The project uses a RAG pipeline to match social media posts with DSM-5 criteria. It utilizes the BGE-M3 embedding model to generate embeddings for both the posts and the criteria, and then uses a FAISS vector database for efficient similarity search. SpanBERT is integrated for advanced filtering and token extraction to improve matching accuracy. The system is optimized for performance on an NVIDIA RTX 3090 GPU.

The project is structured as follows:

-   `src`: Contains the source code for the RAG system, including the RAG pipeline, embedding model, FAISS index, and SpanBERT model.
-   `tests`: Contains the test suite for the project.
-   `data`: Contains the data used in the project, including the DSM-5 criteria and social media posts.
-   `scripts`: Contains scripts for running the RAG system and other tasks.
-   `results`: Contains the output of the RAG system.
-   `logs`: Contains log files.

## Building and Running

The project uses `pip` for dependency management. The dependencies are listed in the `pyproject.toml` and `requirements.txt` files.

**Installation:**

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

**Running the RAG system:**

The main entry point for the RAG system is `scripts/main.py`. It can be run in three modes:

-   `build_index`: Builds a FAISS index from the DSM-5 criteria.
-   `evaluate`: Evaluates social media posts against the DSM-5 criteria.
-   `single_post`: Analyzes a single social media post.

To run the RAG system, use the following command:

```bash
python scripts/main.py --mode <mode> [options]
```

For example, to evaluate 100 posts, run:

```bash
python scripts/main.py --mode evaluate --num_posts 100
```

The `Makefile` also provides a set of useful commands for building, running, and testing the project. For example, to run the tests, use the following command:

```bash
make test
```

## Development Conventions

The project uses `black` for code formatting and `flake8` for linting. The `pyproject.toml` file contains the configuration for these tools. The project also has a comprehensive test suite in the `tests` directory.