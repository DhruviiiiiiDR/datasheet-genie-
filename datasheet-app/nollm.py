import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

st.title("📄 Hybrid Datasheet Retrieval (No LLM)")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
CONFIDENCE_THRESHOLD = 0.6
MAX_ANSWER_CHARS = 120

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_text(file):
    file.seek(0)
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for page_num, page in enumerate(pdf, 1):
        text = page.get_text()
        if text.strip():
            pages.append({"page": page_num, "text": text})
    pdf.close()
    return pages

def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for page_data in pages:
        page_num = page_data["page"]
        words = page_data["text"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = ' '.join(words[i:i + chunk_size])
            if len(chunk_text.split()) > 50:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_id": len(chunks)
                })
    return chunks

def build_faiss_index(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def smart_sentence_extraction(text, question):
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?:])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text[:200]

    q_lower = question.lower()
    tech_keywords = [
        "voltage", "current", "power", "range", "address",
        "i2c", "spi", "size", "bytes", "kb", "flash",
        "eeprom", "sram", "ram", "frequency", "clock", "pin",
        "temperature", "sensitivity", "resolution", "supply"
    ]
    def score_sentence(s):
        score = 0.0
        s_lower = s.lower()
        for kw in tech_keywords:
            if kw in s_lower:
                score += 0.5
        q_words = set(q_lower.split())
        s_words = set(s_lower.split())
        overlap = len(q_words & s_words)
        score += overlap * 0.3
        if re.search(r'\b\d+(\.\d+)?\s*(v|ma|mhz|khz|kb|mb|bytes|°c)\b', s_lower):
            score += 0.7
        if len(s) <= 100:
            score += 0.2
        return score

    sentences.sort(key=score_sentence, reverse=True)
    return sentences[0] if sentences else text[:200]

def retrieve_top_chunks(question, index, chunks, model, top_k=5):
    q_emb = model.encode([question], convert_to_numpy=True)
    q_emb = q_emb.astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        if score < CONFIDENCE_THRESHOLD:
            continue
        results.append({
            "score": float(score),
            "chunk": chunks[idx],
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"]
        })
    return results

uploaded = st.file_uploader("Upload Datasheet PDF", type=["pdf"])

if uploaded:
    pages = extract_pdf_text(uploaded)
    chunks = create_chunks(pages)
    embedder = load_embedder()
    index, embeddings = build_faiss_index(chunks, embedder)

    question = st.text_input("Ask a question about the datasheet:")

    if question:
        with st.spinner("Searching for relevant text..."):
            results = retrieve_top_chunks(question, index, chunks, embedder)
        if not results:
            st.warning("No relevant results found with sufficient confidence.")
        else:
            st.subheader("Answer (extracted from datasheet):")
            best_chunk = results[0]["text"]
            snippet = smart_sentence_extraction(best_chunk, question)
            st.write(snippet)
            st.caption(f"Score: {results[0]['score']:.2f} | Page: {results[0]['page']}")

            with st.expander("Top matched chunks"):
                for res in results:
                    st.markdown(f"**Page {res['page']}**, Score: {res['score']:.2f}")
                    st.text(res['text'][:500] + "…")
