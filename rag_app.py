import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables securely from backend .env file
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Multi PDF - RAG app", layout="wide")

st.title("📄 Multi PDF - RAG app")

# --- Secure Sidebar Configuration (No API Keys Displayed) ---
with st.sidebar:
  st.header("Security & Configuration")

  if os.getenv("OPENAI_API_KEY"):
    st.success("🔒 OpenAI API Key loaded securely from backend.")
  else:
    st.error("❌ OPENAI_API_KEY not found in .env file.")

  langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv(
      "LANGCHAIN_API_KEY"
  )
  if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGCHAIN_PROJECT", "Multi-PDF-RAG-app"
    )
    st.success("📊 LangSmith tracing active.")
  else:
    st.info("ℹ️ LangSmith tracing inactive (Optional).")

  st.divider()
  st.header("Upload PDFs")
  uploaded_files = st.file_uploader(
      "Choose PDF files", type="pdf", accept_multiple_files=True
  )
  process_button = st.button("Process PDFs & Create Vector Store")

# Folder paths & Session State setup
vector_store_path = "My store"
if "vectorstore" not in st.session_state:
  st.session_state.vectorstore = None


@st.cache_resource
def get_embeddings():
  """Initialize OpenAI Embeddings explicitly using 'text-embedding-3-small'"""
  from langchain_openai import OpenAIEmbeddings

  return OpenAIEmbeddings(model="text-embedding-3-small")


def initialize_vector_store(files):
  """Loads multiple uploaded PDFs, chunks them, embeds them into a single FAISS vector store,

  saves it locally as 'My store', and reports total indexed chunks.
  """
  from langchain_community.document_loaders import PyPDFLoader
  from langchain_community.vectorstores import FAISS
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  if not files:
    st.warning("Please upload at least one PDF file before processing.")
    return None

  with st.spinner("Processing uploaded PDFs and building vector store..."):
    all_docs = []

    # Save uploaded files temporarily to parse them with PyPDFLoader
    with tempfile.TemporaryDirectory() as temp_dir:
      for uploaded_file in files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
          f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_file_path)
        all_docs.extend(loader.load())

    if not all_docs:
      st.warning("No valid content found in the uploaded PDFs.")
      return None

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)

    # Embed chunks and create single FAISS vector database
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save vector store locally as "My store"
    vectorstore.save_local(vector_store_path)

    # Store in session state
    st.session_state.vectorstore = vectorstore

    # Report how many chunks were indexed
    st.success(
        f"✅ Vector store successfully built! **Total chunks indexed:**"
        f" {len(chunks)}"
    )
    return vectorstore


# Trigger vector store creation via sidebar button
if process_button:
  if not os.getenv("OPENAI_API_KEY"):
    st.error("Cannot process PDFs: OpenAI API Key is missing in your .env file.")
  else:
    initialize_vector_store(uploaded_files)

# --- Main Query Interface ---
st.subheader("Ask Questions from your Uploaded PDFs")
user_query = st.text_input(
    "Enter your question based on the uploaded documents:"
)

if user_query:
  if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "Please configure your OPENAI_API_KEY inside your backend .env file"
        " before querying."
    )
  else:
    try:
      from langchain_community.vectorstores import FAISS
      from langchain_core.output_parsers import StrOutputParser
      from langchain_core.prompts import ChatPromptTemplate
      from langchain_core.runnables import RunnablePassthrough
      from langchain_openai import ChatOpenAI

      embeddings = get_embeddings()

      # Load vector store into session state if not already loaded
      if st.session_state.vectorstore is None:
        if os.path.exists(vector_store_path):
          st.session_state.vectorstore = FAISS.load_local(
              vector_store_path,
              embeddings,
              allow_dangerous_deserialization=True,
          )
        else:
          st.warning(
              "Vector store not found. Please upload PDFs and click 'Process"
              " PDFs & Create Vector Store'."
          )

      if st.session_state.vectorstore:
        # Configure retriever with similarity search and K=2 across all documents
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )

        # Initialize ChatOpenAI LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Setup RAG Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are an assistant for question-answering tasks. Use the"
                    " following pieces of retrieved context to answer the"
                    " question. If you don't know the answer, say that you"
                    " don't know.\n\n{context}"
                ),
            ),
            ("human", "{input}"),
        ])

        with st.spinner("Searching documents and generating answer..."):
          retrieved_docs = retriever.invoke(user_query)
          context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

          chain = (
              {"context": lambda x: context_text, "input": RunnablePassthrough()}
              | prompt
              | llm
              | StrOutputParser()
          )

          response_answer = chain.invoke(user_query)

        # Display Answer
        st.markdown("### Answer")
        st.write(response_answer)

        # Display Retrieved Source Documents with File Name and Page Number from metadata
        with st.expander("View Retrieved Source Documents (Top K=2 Sources)"):
          for i, doc in enumerate(retrieved_docs):
            source_path = doc.metadata.get("source", "Unknown Source")
            file_name = os.path.basename(source_path)
            # PyPDFLoader stores 0-indexed page numbers; adding 1 for 1-indexed display
            page_num = doc.metadata.get("page", 0) + 1

            st.markdown(f"**Source {i+1}: {file_name} (Page {page_num})**")
            st.write(doc.page_content)
            st.markdown(f"*Full Metadata:* `{doc.metadata}`")
            st.divider()

    except Exception as e:
      st.error(f"An error occurred during execution: {e}")