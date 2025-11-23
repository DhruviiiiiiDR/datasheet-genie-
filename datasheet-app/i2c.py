import streamlit as st
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

# ---------- Page Config ----------
st.set_page_config(
    page_title="Datasheet RAG & I2C Generator",
    page_icon="📄",
    layout="wide"
)

# ---------- Config ----------
CONFIDENCE_THRESHOLD = 0.45
CHUNK_SIZE = 400
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 10
MAX_ANSWER_CHARS = 150

# ---------- Title & Header ----------
st.title("📄 Datasheet RAG & I2C Code Generator")
st.markdown("**Upload a datasheet PDF, ask questions, and auto-generate Arduino I2C driver code**")
st.divider()

# ---------- Helper Functions ----------

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


def build_bm25_index(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def rrf_fusion(faiss_indices, bm25_indices, k=60):
    """Reciprocal Rank Fusion for combining retrieval results"""
    scores = {}
    for rank, idx in enumerate(faiss_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def best_sentence_extraction(text, question):
    if not text:
        return ""
    
    sentences = re.split(r'(?<=[.!?:])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text[:200]
    
    q_lower = question.lower()
    
    def score_sentence(s):
        score = 0.0
        s_lower = s.lower()
        
        # CRITICAL: If question asks about "I2C address", heavily prioritize hex addresses
        if "i2c address" in q_lower or ("address" in q_lower and "i2c" in q_lower):
            # Boost sentences with actual address values
            if re.search(r'(?:address|addr)[\s:=]*(0x[0-9a-fA-F]{2})', s_lower):
                score += 10.0
            # Penalize clock/pin mentions when looking for address
            if re.search(r'(?:clock|scl|sda|sck|mosi|miso|pin)', s_lower):
                score -= 8.0
        
        # Question word overlap
        q_words = set(q_lower.split())
        s_words = set(s_lower.split())
        overlap = len(q_words & s_words)
        score += overlap * 0.5
        
        # Technical keywords
        tech_keywords = [
            "voltage", "current", "power", "range", "address",
            "i2c", "spi", "uart", "size", "bytes", "kb", "flash",
            "eeprom", "sram", "ram", "frequency", "temperature", 
            "sensitivity", "resolution", "supply", "operating", 
            "register", "data", "control", "config"
        ]
        for kw in tech_keywords:
            if kw in s_lower:
                score += 0.4
        
        # Hex addresses boost
        if re.search(r'0x[0-9a-fA-F]{1,4}', s):
            score += 0.7
        
        # Numbers with units
        if re.search(r'\b\d+(\.\d+)?\s*(v|ma|mhz|khz|kb|mb|bytes|°c|hz|ms|a)\b', s_lower):
            score += 0.8
        
        # Brevity bonus
        if len(s) <= 100:
            score += 0.3
        
        return score
    
    sentences.sort(key=score_sentence, reverse=True)
    return sentences[0] if sentences else text[:200]


def smart_trim(text, max_chars=MAX_ANSWER_CHARS):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = max(cut.rfind("."), cut.rfind(":"), cut.rfind(";"))
    if last_period >= int(max_chars * 0.6):
        return cut[:last_period + 1].strip()
    last_space = cut.rfind(" ")
    if last_space > 0:
        return cut[:last_space] + "…"
    return cut + "…"


def hybrid_retrieve_and_answer(question, faiss_index, bm25_index, chunks, model, thresh=CONFIDENCE_THRESHOLD):
    """Hybrid retrieval using FAISS + BM25 with RRF fusion"""
    
    # FAISS semantic search
    q_emb = model.encode([question], convert_to_numpy=True)
    q_emb = q_emb.astype('float32')
    faiss.normalize_L2(q_emb)
    faiss_scores, faiss_indices = faiss_index.search(q_emb, RETRIEVAL_TOP_K)
    
    # BM25 keyword search
    q_tokens = question.lower().split()
    bm25_scores = bm25_index.get_scores(q_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:RETRIEVAL_TOP_K]
    
    # Fuse results with RRF
    fused = rrf_fusion(faiss_indices[0].tolist(), bm25_indices.tolist())
    
    results = []
    for idx, rrf_score in fused[:5]:
        if idx < 0 or idx >= len(chunks):
            continue
        
        # Get original FAISS score
        faiss_score = 0.0
        if idx in faiss_indices[0]:
            pos = np.where(faiss_indices[0] == idx)[0]
            if len(pos) > 0:
                faiss_score = float(faiss_scores[0][pos[0]])
        
        results.append({
            "score": faiss_score,
            "rrf_score": rrf_score,
            "chunk": chunks[idx],
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"]
        })
    
    if not results or results[0]["score"] < thresh:
        return {"answer": "• Answer not found in datasheet.", "results": results}
    
    top = results[0]
    best_sent = best_sentence_extraction(top["text"], question)
    best_sent = smart_trim(best_sent, MAX_ANSWER_CHARS)
    page_ref = f" (pg {top['page']})" if top.get('page') else ""
    answer = f"• {best_sent}{page_ref}"
    
    return {"answer": answer, "results": results}


def extract_i2c_info(chunks):
    """Automatically extract I2C address and registers from all chunks"""
    i2c_addresses = []
    registers = {}
    voltage_info = []
    pin_info = []
    
    # Enhanced patterns - prioritize explicit "I2C address" mentions
    addr_pattern_explicit = r'(?:I2C|i2c)\s*(?:address|addr|slave\s+address)[\s:=]*(0x[0-9a-fA-F]{2})'
    addr_pattern_table = r'(?:address|addr)[\s:]*\|?\s*(0x[0-9a-fA-F]{2})'
    addr_fallback = r'\b0x[0-9a-fA-F]{2}\b'
    reg_pattern = r'(?:register|reg)\s+(\w+)[\s:=]+(0x[0-9a-fA-F]{2,4})'
    voltage_pattern = r'(?:supply|voltage|vcc|vin|operating voltage)[\s:=]*(\d+\.?\d*)\s*(?:v|volt)'
    pin_pattern = r'(?:SCL|SDA|scl|sda)[\s:=]+(\w+)'
    
    for chunk in chunks:
        text = chunk["text"]
        
        # I2C addresses - prioritize explicit mentions
        explicit_matches = re.findall(addr_pattern_explicit, text, re.IGNORECASE)
        if explicit_matches:
            i2c_addresses.extend(explicit_matches)
        
        # Table format addresses
        table_matches = re.findall(addr_pattern_table, text, re.IGNORECASE)
        # Filter out likely pin/clock references
        for addr in table_matches:
            context = text[max(0, text.find(addr)-50):text.find(addr)+50].lower()
            if not any(word in context for word in ['clock', 'scl', 'sda', 'sck', 'pin', 'mosi', 'miso']):
                i2c_addresses.append(addr)
        
        # Fallback only if no explicit addresses found
        if not i2c_addresses:
            fallback_matches = re.findall(addr_fallback, text)
            for addr in fallback_matches:
                context = text[max(0, text.find(addr)-60):text.find(addr)+60].lower()
                if 'address' in context and not any(word in context for word in ['clock', 'scl', 'sda', 'pin']):
                    i2c_addresses.append(addr)
        
        # Registers
        reg_matches = re.findall(reg_pattern, text, re.IGNORECASE)
        for reg_name, reg_addr in reg_matches:
            registers[reg_name.upper()] = reg_addr
        
        # Voltage
        volt_matches = re.findall(voltage_pattern, text, re.IGNORECASE)
        voltage_info.extend(volt_matches)
        
        # Pins
        pin_matches = re.findall(pin_pattern, text)
        pin_info.extend(pin_matches)
    
    # Select most common I2C address
    i2c_address = "0x00"
    if i2c_addresses:
        from collections import Counter
        i2c_address = Counter(i2c_addresses).most_common(1)[0][0]
    
    return {
        "i2c_address": i2c_address,
        "registers": registers,
        "voltage": voltage_info[0] if voltage_info else "3.3",
        "pins": pin_info
    }


def generate_arduino_i2c_code(i2c_info, device_name="Sensor"):
    """Generate Arduino I2C code from extracted info"""
    
    i2c_addr = i2c_info["i2c_address"]
    regs = i2c_info["registers"]
    voltage = i2c_info["voltage"]
    
    reg_config = regs.get("CONFIG", regs.get("CTRL", regs.get("CONTROL", "0x00")))
    reg_data = regs.get("DATA", regs.get("OUT", regs.get("RESULT", "0x01")))
    
    # Generate register definitions
    reg_defines = ""
    for reg_name, reg_addr in list(regs.items())[:5]:
        reg_defines += f"#define REG_{reg_name} {reg_addr}\n"
    
    # Fallback if no registers detected
    if not reg_defines:
        reg_defines = f"#define REG_CONFIG {reg_config}\n#define REG_DATA {reg_data}"
    
    code = f"""// Auto-generated Arduino I2C driver for {device_name}
// I2C Address: {i2c_addr}
// Supply Voltage: {voltage}V

#include <Wire.h>

#define SENSOR_I2C_ADDRESS {i2c_addr}
{reg_defines}

void setup() {{
  Wire.begin();
  Serial.begin(9600);
  delay(100);
  
  Serial.println("Initializing {device_name}...");
  
  // Initialize sensor
  Wire.beginTransmission(SENSOR_I2C_ADDRESS);
  Wire.write({reg_config});
  Wire.write(0x01);  // Enable sensor
  byte error = Wire.endTransmission();
  
  if (error == 0) {{
    Serial.println("{device_name} initialized successfully!");
  }} else {{
    Serial.print("Initialization error: ");
    Serial.println(error);
  }}
}}

void loop() {{
  // Read sensor data
  Wire.beginTransmission(SENSOR_I2C_ADDRESS);
  Wire.write({reg_data});
  byte error = Wire.endTransmission(false);
  
  if (error != 0) {{
    Serial.print("Read error: ");
    Serial.println(error);
    delay(1000);
    return;
  }}
  
  Wire.requestFrom(SENSOR_I2C_ADDRESS, (uint8_t)2);
  
  if (Wire.available() >= 2) {{
    uint16_t rawData = (Wire.read() << 8) | Wire.read();
    
    Serial.print("{device_name} Data: ");
    Serial.print(rawData);
    Serial.print(" (0x");
    Serial.print(rawData, HEX);
    Serial.println(")");
  }} else {{
    Serial.println("Error: Insufficient data from sensor");
  }}
  
  delay(1000);
}}
"""
    return code.strip()


# ---------- Main UI ----------

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Lower = more permissive"
    )
    
    st.divider()
    st.markdown("### 📊 Document Stats")
    if "pages" in st.session_state:
        st.metric("Pages", len(st.session_state["pages"]))
        st.metric("Chunks", len(st.session_state["chunks"]))
    else:
        st.info("Upload a PDF to see stats")
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.caption("• Hybrid search: FAISS + BM25")
    st.caption("• Auto I2C code generation")
    st.caption("• Ask about specs, registers, pins")

# Main content
uploaded = st.file_uploader(
    "📤 Upload Datasheet PDF",
    type=["pdf"],
    help="Upload a sensor or MCU datasheet"
)

if uploaded:
    # Process PDF
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded.name:
        with st.spinner("🔄 Processing PDF with hybrid indexing..."):
            pages = extract_pdf_text(uploaded)
            chunks = create_chunks(pages)
            embedder = load_embedder()
            faiss_index, embeddings = build_faiss_index(chunks, embedder)
            bm25_index = build_bm25_index(chunks)
            i2c_info = extract_i2c_info(chunks)
            
            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = faiss_index
            st.session_state["bm25_index"] = bm25_index
            st.session_state["embeddings"] = embeddings
            st.session_state["embedder"] = embedder
            st.session_state["i2c_info"] = i2c_info
            st.session_state["current_file"] = uploaded.name
        
        st.success(f"✅ Indexed {len(chunks)} chunks from {len(pages)} pages (FAISS + BM25)")
    
    # Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("💬 Ask Questions")
        
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is the I2C address?",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Ask", type="primary", use_container_width=True):
            if question:
                with st.spinner("Searching with hybrid retrieval..."):
                    result = hybrid_retrieve_and_answer(
                        question,
                        st.session_state["faiss_index"],
                        st.session_state["bm25_index"],
                        st.session_state["chunks"],
                        st.session_state["embedder"],
                        thresh=confidence
                    )
                
                st.markdown("#### Answer:")
                if result["answer"] == "• Answer not found in datasheet.":
                    st.warning(result["answer"])
                else:
                    st.success(result["answer"])
                
                if result["results"]:
                    st.caption(f"FAISS Score: {result['results'][0]['score']:.2f} | RRF Score: {result['results'][0]['rrf_score']:.3f}")
                    
                    with st.expander("📚 View source chunks"):
                        for i, res in enumerate(result["results"], 1):
                            st.markdown(f"**Chunk {i}** (Page {res['page']}, Score: {res['score']:.2f})")
                            st.text(res['text'][:400] + "…")
                            st.divider()
    
    with col_right:
        st.subheader("⚙️ Arduino I2C Code")
        
        i2c_info = st.session_state["i2c_info"]
        
        st.markdown("**Detected Info:**")
        st.code(f"I2C Address: {i2c_info['i2c_address']}\nRegisters: {len(i2c_info['registers'])}\nVoltage: {i2c_info['voltage']}V", language="text")
        
        device_name = st.text_input(
            "Device name:",
            value="Sensor",
            help="Used in code comments"
        )
        
        if st.button(
            "🔧 Generate Arduino Code",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Generating code..."):
                arduino_code = generate_arduino_i2c_code(i2c_info, device_name=device_name)
            
            st.success("✅ Code generated!")
            
            with st.expander("📄 View Generated Code", expanded=True):
                st.code(arduino_code, language="arduino", line_numbers=True)
            
            st.download_button(
                "⬇️ Download .ino file",
                arduino_code,
                file_name=f"{device_name.lower().replace(' ', '_')}_i2c.ino",
                mime="text/x-arduino",
                use_container_width=True
            )

else:
    st.info("👆 Upload a PDF datasheet to get started")
    
    st.markdown("---")
    st.markdown("### Features:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔍 Hybrid Search")
        st.caption("FAISS + BM25 with RRF fusion for better accuracy")
    
    with col2:
        st.markdown("#### 💬 Smart Q&A")
        st.caption("Ask about specs, pins, registers")
    
    with col3:
        st.markdown("#### ⚙️ Auto Code Gen")
        st.caption("Instant Arduino I2C driver from datasheet")

st.divider()
st.caption("Built with Streamlit • FAISS + BM25 Hybrid Retrieval")
