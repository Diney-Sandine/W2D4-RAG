# 📄 Multi PDF - RAG Application

A powerful, single-file Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, and **FAISS**, powered by **OpenAI** (`text-embedding-3-small` and `gpt-4o-mini`). This app allows users to upload multiple PDF documents at once, embed them into a single local vector store, and ask precise questions with exact source attributions (file name and page numbers).

---

## 🚀 Features

- **Multi-PDF Upload:** Upload multiple PDF documents simultaneously via a single clean interface.
- **Unified Vector Database:** Automatically chunks and embeds all uploaded documents into a single local FAISS vector store saved locally as `"My store"`.
- **Chunk Counting Feedback:** Reports the exact number of indexed chunks upon successful processing[cite: 1].
- **Precise Source Attribution:** Retrieves top-$k$ ($k=2$) results and displays exact source file names and 1-indexed page numbers pulled directly from chunk metadata[cite: 1].
- **Secure Backend Configuration:** Keeps credentials secure using a `.env` file without exposing API keys in the frontend UI[cite: 1].
- **Session State Management:** Maintains vector store states efficiently using Streamlit session state[cite: 1].

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit[cite: 1]
- **Orchestration:** LangChain, LangChain Expression Language (LCEL)[cite: 1]
- **Vector Database:** FAISS[cite: 1]
- **Embeddings & LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)[cite: 1]
- **PDF Parsing:** PyPDFLoader / pypdf[cite: 1]
- **Environment Management:** python-dotenv[cite: 1]

---

## 📂 Project Structure

```text
W2D4 RAG/
│
├── .streamlit/
│   └── config.toml         # Streamlit configuration (optional)
├── PDFs/                   # Folder for local PDF documents
├── My store/               # Local FAISS vector database store
├── .env                    # Backend environment variables (API keys)
├── rag_app.py              # Main Streamlit application script
└── README.md               # Project documentation

