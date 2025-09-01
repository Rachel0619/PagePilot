# PagePilot

A multi-modal RAG (Retrieval-Augmented Generation) application that allows you to upload PDF documents and ask questions about their content. The system can process and understand text, tables, and images within PDFs to provide comprehensive answers.

## Technology Stack

- Frontend: Streamlit
- PDF parsing: unstructured
- LLMs: llama3.1:8b for texts and tables, gpt-4o-mini for images
- Vector Database: Chroma
- Embeddings: OpenAIEmbeddings
- Framework: LangChain

## How to use it

### Prerequisites

- Python 3.11+
- uv package manager

### Get started

1. Clone the repo

```
git clone git@github.com:Rachel0619/PagePilot.git
cd PagePilot
```

2. Install dependencies

```
uv sync
```

3. Set up environment variables

copy `.env.example` and rename it to `.env`, then replace templated API keys with your own.

4. Install Ollama

```
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
```

5. Run the app

```
uv run streamlit run app.py
```

6. Ask questions

Navigate to http://localhost:8501, upload your pdf and start asking questions.