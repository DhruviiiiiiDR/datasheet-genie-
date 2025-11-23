import streamlit as st
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import ollama

st.title("📄 Hybrid Datasheet Search with RAGAS")

CONFIDENCE_THRESHOLD = 0.3
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

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
        score += len(q_words & s_words) * 0.3
        if re.search(r'\b\d+(\.\d+)?\s*(v|ma|mhz|khz|kb|mb|bytes|°c)\b', s_lower):
            score += 0.7
        if len(s) <= 100:
            score += 0.2
        return score
    sentences.sort(key=score_sentence, reverse=True)
    return sentences[0] if sentences else text[:200]

def retrieve_top_chunks(question, index, chunks, model, top_k=10):
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

def find_i2c_address(chunks):
    pattern = r'0x[0-9a-fA-F]{2}'
    for chunk in chunks:
        matches = re.findall(pattern, chunk['text'])
        if matches:
            return matches[0]
    return "I2C address not found in datasheet."

def generate_arduino_code_template(chunks):
    i2c_address = "0x00"
    key_registers = {"REG_CONFIG": "0x00", "REG_DATA": "0x00"}
    address_pattern = r'0x[0-9A-Fa-f]{2}'
    for chunk in chunks:
        matches = re.findall(address_pattern, chunk["text"])
        if matches:
            i2c_address = matches[0]
            break
    reg_pattern = r'(?:register|reg)\s*(\w*)\s*[:=]?\s*(0x[0-9A-Fa-f]{2})'
    for chunk in chunks:
        regs = re.findall(reg_pattern, chunk["text"], flags=re.IGNORECASE)
        if regs:
            for reg_name, reg_addr in regs:
                key_registers[reg_name.upper() or "REG_UNKNOWN"] = reg_addr
            break
    code_template = f"""
#include <Wire.h>

#define SENSOR_I2C_ADDRESS {i2c_address}
#define REG_CONFIG {key_registers.get('REG_CONFIG', '0x00')}
#define REG_DATA {key_registers.get('REG_DATA', '0x00')}

void setup() {{
  Wire.begin();
  Serial.begin(9600);
  // Initialize sensor configuration
  Wire.beginTransmission(SENSOR_I2C_ADDRESS);
  Wire.write(REG_CONFIG);
  Wire.write(0x01); // Sample config value
  Wire.endTransmission();
}}

void loop() {{
  Wire.beginTransmission(SENSOR_I2C_ADDRESS);
  Wire.write(REG_DATA);
  Wire.endTransmission();
  
  Wire.requestFrom(SENSOR_I2C_ADDRESS, 2);
  if (Wire.available() == 2) {{
    int data = Wire.read() << 8 | Wire.read();
    Serial.print("Sensor Data: ");
    Serial.println(data);
  }}
  
  delay(1000);
}}
"""
    return code_template.strip()

def generate_ragas_answer(question, context_chunks, model_name="qwen2.5:3b"):
    context_parts = []
    for chunk in context_chunks[:5]:
        snippet = smart_sentence_extraction(chunk["text"], question)
        page = chunk["page"]
        context_parts.append(f"[Page {page}] {snippet}")
    context = "\n\n".join(context_parts)
    prompt = f"""Answer the question based strictly on the following datasheet excerpts with page references.

Context:
{context}

Question:
{question}

Please provide a concise and anchored answer citing the relevant page numbers.
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 200}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"LLM query error: {str(e)}"

uploaded = st.file_uploader("Upload Datasheet PDF", type=["pdf"])

if uploaded:
    st.info("Extracting text...")
    pages = extract_pdf_text(uploaded)
    chunks = create_chunks(pages)
    embedder = load_embedder()
    index, embeddings = build_faiss_index(chunks, embedder)

    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Searching for relevant data..."):
            results = retrieve_top_chunks(question, index, chunks, embedder)

        if not results:
            fallback = find_i2c_address(chunks)
            st.warning("No relevant results found with sufficient confidence.")
            st.info(f"Fallback I2C address (regex search): {fallback}")
        else:
            st.subheader("Answer (excerpt from datasheet):")
            best_text = results[0]["text"]
            snippet = smart_sentence_extraction(best_text, question)
            st.write(snippet)
            st.caption(f"Score: {results[0]['score']:.2f} | Page: {results[0]['page']}")

            with st.expander("Top matched text chunks"):
                for res in results:
                    st.markdown(f"**Page {res['page']}** — Score: {res['score']:.2f}")
                    st.text(res['text'][:500] + "…")

            if st.button("Generate RAGAS Answer (Qwen 2.5:3b)"):
                with st.spinner("Generating RAGAS anchored answer..."):
                    ragas_answer = generate_ragas_answer(question, results)
                st.markdown("### 🧠 RAGAS Generated Answer")
                st.write(ragas_answer)

    if st.button("Generate Arduino I2C Code (Template)"):
        arduino_code = generate_arduino_code_template(chunks)
        st.code(arduino_code, language="arduino")
        st.download_button("Download Arduino Code", data=arduino_code, file_name="sensor_driver.ino", mime="text/x-arduino")

else:
    st.info("Upload a PDF datasheet to get started.")
