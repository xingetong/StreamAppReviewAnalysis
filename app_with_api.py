import re
import ast
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import parser as date_parser
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os
import requests
import time


# ============================================
# CONFIGURATION
# ============================================

def get_hf_token():
    try:
        # Try to get from Streamlit secrets
        return st.secrets["HUGGINGFACE_TOKEN"]["token"]
    except (KeyError, FileNotFoundError):
        # Fallback to environment variable
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            st.error("⚠️ HuggingFace token not found! Please add it to Streamlit secrets or as an environment variable.")
            st.stop()
        return token

# ============================================
# DATE PARSING FUNCTIONS
# ============================================
def parse_time_column(df, time_col='time'):
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not in DataFrame")
    if np.issubdtype(df[time_col].dtype, np.datetime64):
        return df
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    return df


def interpret_date_filter_from_question(question, reference_date=None):
    q = (question or '').lower()
    if reference_date is None:
        reference_date = datetime.now()
    ref = reference_date

    m = re.search(r'from\s+([^,]+?)\s+to\s+([^,]+)', q)
    if m:
        try:
            s = date_parser.parse(m.group(1), dayfirst=True, default=ref)
            e = date_parser.parse(m.group(2), dayfirst=True, default=ref)
            start = datetime(s.year, s.month, s.day)
            end = datetime(e.year, e.month, e.day) + timedelta(days=1) - timedelta(seconds=1)
            return start, end
        except Exception:
            pass

    m = re.search(r'last\s+(\d+)\s+days?', q)
    if m:
        n = int(m.group(1))
        end = ref
        start = ref - timedelta(days=n)
        return datetime(start.year, start.month, start.day), datetime(end.year, end.month, end.day, 23,59,59)

    m = re.search(r'last\s+(\d+)\s+weeks?', q)
    if m:
        n = int(m.group(1))
        end = ref
        start = ref - timedelta(weeks=n)
        return datetime(start.year, start.month, start.day), datetime(end.year, end.month, end.day, 23,59,59)

    if any(x in q for x in ['recent', 'recently', 'past month']):
        end = ref
        start = ref - relativedelta(days=30)
        return datetime(start.year, start.month, start.day), datetime(end.year, end.month, end.day, 23,59,59)

    if 'last week' in q:
        end = ref
        start = ref - timedelta(days=7)
        return datetime(start.year, start.month, start.day), datetime(end.year, end.month, end.day, 23,59,59)

    m = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b(?:\s+(\d{4}))?', q)
    if m:
        month_str = m.group(1).capitalize()
        year = int(m.group(2)) if m.group(2) else ref.year
        month = datetime.strptime(month_str, '%B').month
        start = datetime(year, month, 1)
        next_month = start + relativedelta(months=1)
        end = next_month - timedelta(seconds=1)
        return start, end

    m = re.search(r'(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})', q)
    if m:
        try:
            d = date_parser.parse(m.group(1), dayfirst=True)
            start = datetime(d.year, d.month, d.day)
            end = start + timedelta(days=1) - timedelta(seconds=1)
            return start, end
        except Exception:
            pass

    return None, None


def filter_df_by_date(df, start=None, end=None, time_col='time'):
    df = parse_time_column(df, time_col=time_col)
    if start is None and end is None:
        return df
    if start is None:
        start = df[time_col].min()
    if end is None:
        end = df[time_col].max()
    mask = (df[time_col] >= pd.Timestamp(start)) & (df[time_col] <= pd.Timestamp(end))
    return df.loc[mask].copy()


# ============================================
# DOCUMENT PROCESSING
# ============================================
def build_documents(df, text_col="correctmapping_nosym", topic_col="predicted_topic", 
                   sentiment_col="predicted_sentiment", time_col="time"):
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "topic": str(row[topic_col]) if topic_col in df.columns else None,
            "sentiment": str(row[sentiment_col]) if sentiment_col in df.columns else None,
            "time": str(row[time_col]) if time_col in df.columns else None
        }
        doc = Document(page_content=str(row[text_col]), metadata=metadata)
        documents.append(doc)
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def filter_chunks_by_date(chunks, start, end):
    filtered = []
    for doc in chunks:
        try:
            if "time" in doc.metadata and doc.metadata["time"]:
                doc_time = datetime.strptime(doc.metadata["time"], "%d/%m/%Y")
                if start <= doc_time <= end:
                    filtered.append(doc)
        except Exception:
            continue
    return filtered


# ============================================
# LLM via HuggingFace Inference API
# ============================================
class HuggingFaceInferenceAPI:
    """Use HuggingFace Inference API instead of loading model locally"""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", token=None):
        self.model_name = model_name
        self.token = token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def invoke(self, prompt, max_tokens=512, temperature=0.7):
        """Call HuggingFace Inference API"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        # Retry logic for model loading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "")
                    return str(result)
                
                elif response.status_code == 503:
                    # Model is loading
                    if attempt < max_retries - 1:
                        wait_time = 20 * (attempt + 1)
                        st.warning(f"🔄 Model is loading... waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "⚠️ Model is still loading. Please try again in a moment."
                
                else:
                    error_msg = response.json().get("error", "Unknown error")
                    return f"❌ API Error: {error_msg}"
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"⏱️ Request timeout, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(10)
                    continue
                return "⚠️ Request timeout. Please try again."
            
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        return "⚠️ Failed after multiple attempts. Please try again later."


# ============================================
# MODEL LOADING (CACHED)
# ============================================
@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    """Load HuggingFace Inference API client"""
    token = get_hf_token()
    return HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        token=token
    )


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    return pd.read_csv("data/review_dataset.csv")


# ============================================
# RAG FUNCTIONS
# ============================================
def build_retriever(chunks, embedding_model, k=5):
    """Build FAISS retriever from chunks"""
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def answer_question(question, chunks, embedding_model, llm):
    """Answer question using RAG"""
    # 1. Parse date filter
    start, end = interpret_date_filter_from_question(question, reference_date=datetime.now())

    # 2. Apply date filter
    if start and end:
        filtered_chunks = filter_chunks_by_date(chunks, start, end)
        date_info = f"(Filtered: {start.date()} to {end.date()})"
    else:
        filtered_chunks = chunks
        date_info = "(All dates)"

    if not filtered_chunks:
        return f"No reviews found in the selected date range. {date_info}"

    # 3. Build retriever
    retriever = build_retriever(filtered_chunks, embedding_model, k=5)

    # 4. Retrieve docs
    retrieved_docs = retriever.get_relevant_documents(question)

    # 5. Build context
    context = "\n\n".join([f"Review {i+1}: {d.page_content[:300]}" 
                           for i, d in enumerate(retrieved_docs)])
    
    # 6. Create prompt for Mistral
    prompt = f"""<s>[INST] You are a helpful assistant summarizing player feedback for game developers.

Question: {question}
Date Range: {date_info}

Relevant reviews:
{context}

Please provide a concise summary with:
1. Main complaints (3-5 bullet points)
2. Key praises (if any, 2-3 bullet points)
3. Brief overall summary

Be specific and factual. [/INST]"""
    
    # 7. Get LLM response
    response = llm.invoke(prompt, max_tokens=512)
    
    return response


# ============================================
# STREAMLIT UI
# ============================================
def main():
    st.set_page_config(page_title="Game Review Q&A", page_icon="🎮", layout="wide")
    
    st.title("🎮 Game Review Q&A System")
    st.write("Ask questions about player complaints, praises, or topics from reviews.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("🚀 Using Mistral-7B via HuggingFace Inference API")
        st.success("✅ No local model loading required!")
        
        st.markdown("---")
        st.markdown("**Benefits:**")
        st.markdown("- ✅ Works on Streamlit Cloud")
        st.markdown("- ✅ No RAM limits")
        st.markdown("- ✅ Fast inference")
        st.markdown("- ⚠️ First request may take 20s (model loading)")
    
    # Load data and models
    try:
        with st.spinner("Loading data and initializing..."):
            df = load_data()
            embedding_model = load_embedding_model()
            llm = load_llm()
            
            # Build documents and chunks (cached in session state)
            if 'chunks' not in st.session_state:
                with st.spinner("Processing documents..."):
                    documents = build_documents(df)
                    st.session_state.chunks = chunk_documents(documents)
            
            chunks = st.session_state.chunks
            
        st.success(f"✅ Loaded {len(df)} reviews and {len(chunks)} document chunks")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Question input
    st.subheader("💬 Ask Your Question")
    
    example_questions = [
        "What were the most common complaints in March 2025?",
        "What do players say about gameplay in the last 30 days?",
        "Show me recent positive feedback",
        "What are the issues with graphics?"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Enter your question:", 
            placeholder="e.g., What were the main complaints last week?"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    
    with st.expander("💡 Try these example questions"):
        for i, example in enumerate(example_questions):
            if st.button(example, key=f"example_{i}"):
                user_question = example
                st.rerun()
    
    # Answer
    if ask_button or user_question:
        if user_question:
            with st.spinner("🤔 Analyzing reviews and generating answer..."):
                try:
                    response = answer_question(user_question, chunks, embedding_model, llm)
                    
                    st.subheader("📝 Answer")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.exception(e)
        else:
            st.warning("Please enter a question first.")
    
    # Optional: Show data preview
    with st.expander("📊 View Dataset Preview"):
        st.dataframe(df.head(10))


if __name__ == "__main__":
    main()