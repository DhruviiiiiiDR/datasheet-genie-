import streamlit as st
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from collections import Counter


# ---------- Page Config ----------
st.set_page_config(
    page_title="📄 Sensor Datasheet Assistant",
    page_icon="📄",
    layout="centered"
)

# Minimal clean UI styling
st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    .block-container {max-width: 800px; padding-top: 1rem;}
    h1 {text-align:center; font-size: 2.4rem;}
    </style>
""", unsafe_allow_html=True)


# ---------- Config ----------
CONFIDENCE_THRESHOLD = 0.42
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 15
MAX_ANSWER_CHARS = 180


# =====================================================================
#                     Helper Functions (unchanged)
# =====================================================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(file):
    try:
        file.seek(0)
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        pages = []
        for page_num, page in enumerate(pdf, 1):
            text = page.get_text()
            if text.strip():
                pages.append({"page": page_num, "text": text})
        pdf.close()
        return pages
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []


def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for page_data in pages:
        page_num, text = page_data["page"], page_data["text"]
        sentences = re.split(r'(?<=[.!?:])\s+|\n{2,}', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            length = len(words)

            if current_length + length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) > 30:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "chunk_id": len(chunks)
                    })

                overlap_words = " ".join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_length += length

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) > 30:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_id": len(chunks)
                })

    return chunks


def build_faiss_index(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def build_bm25_index(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def advanced_rrf_fusion(faiss_results, bm25_results, k=60, faiss_weight=0.6, bm25_weight=0.4):
    scores = {}
    for rank, (idx, score) in enumerate(faiss_results):
        scores[idx] = scores.get(idx, 0) + faiss_weight * (1 / (k + rank + 1)) + (score * 0.18)

    for rank, (idx, score) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + bm25_weight * (1 / (k + rank + 1)) + (min(score / 10, 1) * 0.08)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def extract_entities(text):
    return {
        'addresses': re.findall(r'0x[0-9a-fA-F]{2,4}', text),
        'voltages': re.findall(r'\d+\.?\d*\s*[vV]', text),
        'frequencies': re.findall(r'\d+\.?\d*\s*(MHz|kHz|Hz)', text, re.I)
    }


def determine_question_type(question):
    q = question.lower()
    if "i2c" in q and "address" in q: return "i2c_address"
    if "voltage" in q or "vcc" in q: return "voltage"
    if "frequency" in q: return "frequency"
    if "register" in q: return "register"
    if "pin" in q: return "pin"
    return "general"


def score_sentence_advanced(sentence, question, qtype):
    score = 0
    s = sentence.lower()
    q = question.lower()

    patterns = {
        "i2c_address": (["i2c address", "slave address"], 15),
        "voltage": (["supply voltage", "vcc"], 8),
        "frequency": (["clock", "frequency"], 8),
        "register": (["register", "reg"], 10),
        "pin": (["scl", "sda", "pin"], 8),
    }

    if qtype in patterns:
        keywords, w = patterns[qtype]
        for kw in keywords:
            if kw in s:
                score += w

    score += len(set(s.split()) & set(q.split())) * 1.2

    if re.search(r'0x[0-9a-fA-F]{2}', s):
        score += 2

    return score


def best_sentence_extraction(text, question):
    sentences = re.split(r'(?<=[.!?:])\s+|\n+', text)
    sentences = [s for s in sentences if len(s) > 10]

    qtype = determine_question_type(question)
    scored = [(score_sentence_advanced(s, question, qtype), s) for s in sentences]
    scored.sort(reverse=True)

    return scored[0][1] if scored else text[:200]


def smart_trim(text, max_len=MAX_ANSWER_CHARS):
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_dot = cut.rfind(".")
    if last_dot > max_len * 0.4:
        return cut[:last_dot + 1]
    return cut + "…"


def hybrid_retrieve_and_answer(question, faiss_index, bm25_index, chunks, model):
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    faiss_scores, faiss_indices = faiss_index.search(q_emb, RETRIEVAL_TOP_K)
    faiss_results = [(int(idx), float(score)) for idx, score in zip(faiss_indices[0], faiss_scores[0])]

    q_tokens = question.lower().split()
    bm25_scores = bm25_index.get_scores(q_tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:RETRIEVAL_TOP_K]
    bm25_results = [(int(i), float(bm25_scores[i])) for i in bm25_top]

    fused = advanced_rrf_fusion(faiss_results, bm25_results)
    best_idx = fused[0][0]
    best_chunk = chunks[best_idx]

    best_sentence = best_sentence_extraction(best_chunk["text"], question)
    answer = smart_trim(best_sentence) + f" (pg {best_chunk['page']})"

    return {"answer": answer, "entities": extract_entities(best_sentence)}


@st.cache_data
def extract_i2c_info(chunks):
    addrs = []
    regs = {}
    voltages = []

    for c in chunks:
        text = c["text"]

        addrs.extend(re.findall(r'0x[0-9a-fA-F]{2}', text))

        m = re.findall(r'(\w+)\s*[:=]\s*(0x[0-9A-Fa-f]{2,4})', text)
        for name, addr in m:
            regs[name.upper()] = addr

        v = re.findall(r'(\d+\.?\d*)\s*V', text)
        voltages.extend(v)

    i2c_address = Counter(addrs).most_common(1)[0][0] if addrs else "0x00"
    voltage = Counter(voltages).most_common(1)[0][0] if voltages else "3.3"

    return {"i2c_address": i2c_address, "registers": regs, "voltage": voltage}


def generate_arduino_i2c_code(i2c_info, device_name="Sensor"):
    addr = i2c_info["i2c_address"]
    regs = i2c_info["registers"]

    reg_defines = ""
    for name, val in list(regs.items())[:8]:
        clean = name.upper().replace(" ", "_")
        reg_defines += f"#define REG_{clean} {val}\n"

    code = f"""
#include <Wire.h>

#define I2C_ADDR {addr}

{reg_defines}

void writeReg(uint8_t reg, uint8_t value) {{
    Wire.beginTransmission(I2C_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}}

uint8_t readReg(uint8_t reg) {{
    Wire.beginTransmission(I2C_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(I2C_ADDR, 1);
    return Wire.read();
}}

void setup() {{
    Wire.begin();
    Serial.begin(115200);
}}

void loop() {{
    uint8_t data = readReg(0x00);
    Serial.println(data, HEX);
    delay(500);
}}
"""
    return code


# =====================================================================
#                              UI (Minimal)
# =====================================================================

st.title("📄 Sensor Datasheet Assistant")

uploaded = st.file_uploader("Upload a Sensor Datasheet (PDF)", type=["pdf"])

if uploaded:
    pages = extract_pdf_text(uploaded)
    chunks = create_chunks(pages)
    model = load_embedder()

    faiss_index, _ = build_faiss_index(chunks, model)
    bm25_index = build_bm25_index(chunks)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question:")

    if question:
        result = hybrid_retrieve_and_answer(question, faiss_index, bm25_index, chunks, model)
        st.subheader("Answer")
        st.write(result["answer"])

    if st.button("Generate Arduino I2C Code"):
        i2c_info = extract_i2c_info(chunks)
        code = generate_arduino_i2c_code(i2c_info)
        st.subheader("Arduino I2C Code")
        st.code(code, language="cpp")
