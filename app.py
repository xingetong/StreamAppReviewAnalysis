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
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
import streamlit as st

from huggingface_hub import login
login(token="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")


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


def top_topics_by_sentiment(df, topic_col='topic_label', sentiment_col='sentiment', top_k=10):
    total = df.groupby(topic_col).size().rename('total_count')
    neg = df[df[sentiment_col]=='negative'].groupby(topic_col).size().rename('negative_count')
    pos = df[df[sentiment_col]=='positive'].groupby(topic_col).size().rename('positive_count')
    neu = df[df[sentiment_col]=='neutral'].groupby(topic_col).size().rename('neutral_count')

    agg = pd.concat([total, neg, pos, neu], axis=1).fillna(0).astype(int)
    agg['negative_pct'] = (agg['negative_count'] / agg['total_count'] * 100).round(1)
    agg['positive_pct'] = (agg['positive_count'] / agg['total_count'] * 100).round(1)
    agg = agg.sort_values('negative_count', ascending=False)
    return agg.head(top_k)

def summarize_aspect(df, aspect, topic_col='topic_label', text_col='text', sentiment_col='sentiment', top_examples=5):
    if topic_col in df.columns:
        sub = df[df[topic_col].astype(str).str.contains(aspect, case=False, na=False)]
    else:
        sub = df[df[text_col].astype(str).str.contains(aspect, case=False, na=False)]
    if sub.empty:
        return {'aspect': aspect, 'count': 0, 'message': 'No matching reviews found.'}
    counts = sub[sentiment_col].value_counts().to_dict()
    total = len(sub)
    examples = sub.sort_values(by='time', ascending=False)[text_col].head(top_examples).tolist()
    return {'aspect': aspect, 'total': total, 'sentiment_breakdown': counts, 'examples': examples}

def _load_embeddings_from_column(df, embedding_col='embedding'):
    if embedding_col not in df.columns:
        return None
    first_nonnull = df[embedding_col].dropna().iloc[0]
    if isinstance(first_nonnull, str):
        arr = df[embedding_col].apply(lambda x: np.array(ast.literal_eval(x)) if pd.notnull(x) else None)
        data = np.stack(arr.dropna().to_list()) if arr.dropna().size>0 else np.empty((0,0))
        return data, arr
    elif isinstance(first_nonnull, (list, np.ndarray)):
        arr = df[embedding_col].apply(lambda x: np.array(x) if pd.notnull(x) else None)
        data = np.stack(arr.dropna().to_list()) if arr.dropna().size>0 else np.empty((0,0))
        return data, arr
    else:
        return None

def retrieve_top_k_from_subset(query_embedding, df_subset, embeddings_subset, k=10):
    sims = cosine_similarity([query_embedding], embeddings_subset)[0]
    top_idx = np.argsort(sims)[-min(k, len(sims)):][::-1]
    results = df_subset.reset_index(drop=True).iloc[top_idx].copy()
    results['score'] = sims[top_idx]
    return results

def answer_question_with_date_filter(question, df, embedding_fn, llm_fn,
                                     time_col='time', topic_col='topic_label',
                                     sentiment_col='sentiment', embedding_col='embedding', top_k=6):
    start, end = interpret_date_filter_from_question(question, reference_date=datetime.now())
    df_filtered = filter_df_by_date(df, start, end, time_col=time_col)

    m = re.search(r'about\s+([a-zA-Z0-9_\- "]+)', question, re.IGNORECASE)
    aspect = m.group(1).strip(' "') if m else None

    try:
        top_topics = top_topics_by_sentiment(df_filtered, topic_col=topic_col, sentiment_col=sentiment_col, top_k=top_k)
    except Exception:
        top_topics = pd.DataFrame()

    if aspect:
        aspect_summary = summarize_aspect(df_filtered, aspect, topic_col=topic_col, text_col='text', sentiment_col=sentiment_col)
    else:
        aspect_summary = None

    embeddings_info = _load_embeddings_from_column(df_filtered, embedding_col=embedding_col)
    retrieved_docs = pd.DataFrame()
    if embeddings_info is not None:
        embeddings_array, _ = embeddings_info
        if embeddings_array.size>0:
            query_emb = embedding_fn(question)
            retrieved_docs = retrieve_top_k_from_subset(query_emb, df_filtered.reset_index(drop=True), embeddings_array, k=10)

    human_date_range = f"{start.date() if start else 'ALL'} to {end.date() if end else 'ALL'}"
    top_table = top_topics.to_string() if not top_topics.empty else 'No topics available.'
    examples = '\n'.join([f"- {t[:300]}" for t in (retrieved_docs['text'].tolist() if 'text' in retrieved_docs.columns else [])])

    system_prompt = "You are an assistant that summarises player reviews for game developers. Be concise and factual."
    user_prompt = f"""
Question: {question}
Date filter: {human_date_range}
Top topics table (negative-focused):
{top_table}

Top review excerpts:
{examples}

Please:
1) List 3 bullets of the main complaints in the date range (topic + short reason) if any.
2) List up to 2 praises if present.
3) Provide a small table of the top 5 topics with total and negative_pct.
4) If the user asked about an aspect ({aspect}), include counts and up to 3 short recent examples.
"""
    llm_answer = llm_fn(system_prompt, user_prompt)

    return {
        'llm_answer': llm_answer,
        'aggregates': top_topics,
        'aspect_summary': aspect_summary,
        'retrieved_docs': retrieved_docs
    }


def build_documents(df, text_col="correctmapping_nosym", topic_col="predicted_topic", sentiment_col="predicted_sentiment", time_col="time"):
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

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_retriever(chunks, k=5):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": k})

@st.cache(allow_output_mutation=True)
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)


def llm_fn(system_prompt, user_prompt):
    full_prompt = system_prompt + "\n\n" + user_prompt
    return llm.invoke(full_prompt)

def answer_question(question, chunks):
    # 1. Parse date filter
    start, end = interpret_date_filter_from_question(question, reference_date=datetime.now())

    # 2. Apply date filter
    if start and end:
        filtered_chunks = filter_chunks_by_date(chunks, start, end)
    else:
        filtered_chunks = chunks

    if not filtered_chunks:
        return "No reviews found in the selected date range."

    # 3. Build retriever
    retriever = build_retriever(filtered_chunks, k=5)

    # 4. Retrieve docs
    retrieved_docs = retriever.get_relevant_documents(question)

    # 5. Summarize with LLM
    context = "\n".join([d.page_content for d in retrieved_docs])
    system_prompt = "You are a helpful assistant summarizing player feedback for game developers."
    user_prompt = f"Question: {question}\n\nRelevant reviews:\n{context}\n\nSummarize key complaints and praises."
    return llm_fn(system_prompt, user_prompt)

@st.cache_data
def load_data():
    return pd.read_csv("data/review_dataset.csv")


st.title("Game Review Q&A")
st.write("Ask questions about player complaints, praises, or topics from reviews.")

df = load_data()
documents = build_documents(df)
chunks = chunk_documents(documents)
retriever = build_retriever(chunks)
llm = load_llm()

user_question = st.text_input("Enter your question", "What were the most common complaints in March 2025?")

if user_question:
    retrieved_docs = retriever.get_relevant_documents(user_question)
    context = "\n".join([d.page_content for d in retrieved_docs])
    system_prompt = "You are a helpful assistant summarizing player feedback for game developers."
    user_prompt = f"Question: {user_question}\n\nRelevant reviews:\n{context}\n\nSummarize key complaints and praises."
    response = llm.invoke(system_prompt + "\n\n" + user_prompt)
    st.subheader("Answer")
    st.write(response)
