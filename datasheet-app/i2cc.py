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
    page_title="Datasheet RAG & I2C Generator",
    page_icon="📄",
    layout="wide"
)

# ---------- Config ----------
CONFIDENCE_THRESHOLD = 0.42
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 15
MAX_ANSWER_CHARS = 180

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
    """Improved chunking with sentence boundaries"""
    chunks = []
    
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?:])\s+|\n{2,}', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) > 30:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "chunk_id": len(chunks)
                    })
                
                # Start new chunk with overlap
                overlap_words = ' '.join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) > 30:
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


def advanced_rrf_fusion(faiss_results, bm25_results, k=60, faiss_weight=0.6, bm25_weight=0.4):
    """Weighted RRF fusion with configurable weights"""
    scores = {}
    
    # Add FAISS results with higher weight
    for rank, (idx, score) in enumerate(faiss_results):
        rrf_score = faiss_weight * (1 / (k + rank + 1))
        semantic_bonus = faiss_weight * (score * 0.3)  # Bonus for high semantic similarity
        scores[idx] = scores.get(idx, 0) + rrf_score + semantic_bonus
    
    # Add BM25 results
    for rank, (idx, score) in enumerate(bm25_results):
        rrf_score = bm25_weight * (1 / (k + rank + 1))
        keyword_bonus = bm25_weight * (min(score / 10, 1) * 0.2)  # Normalize and cap bonus
        scores[idx] = scores.get(idx, 0) + rrf_score + keyword_bonus
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def extract_entities(text):
    """Extract key technical entities from text"""
    entities = {
        'addresses': re.findall(r'0x[0-9a-fA-F]{2,4}', text),
        'voltages': re.findall(r'\d+\.?\d*\s*[vV](?:olt)?', text),
        'frequencies': re.findall(r'\d+\.?\d*\s*(?:MHz|KHz|Hz|kHz|mhz)', text, re.IGNORECASE),
        'currents': re.findall(r'\d+\.?\d*\s*(?:mA|uA|A|µA)', text),
        'temperatures': re.findall(r'-?\d+\.?\d*\s*°?[CcFf]', text),
        'sizes': re.findall(r'\d+\.?\d*\s*(?:KB|MB|GB|bytes|bits)', text, re.IGNORECASE),
        'ranges': re.findall(r'\d+\.?\d*\s*(?:to|-|–)\s*\d+\.?\d*', text)
    }
    return entities


def score_sentence_advanced(sentence, question, question_type):
    """Enhanced sentence scoring based on question type"""
    score = 0.0
    s_lower = sentence.lower()
    q_lower = question.lower()
    
    # Question type specific patterns
    patterns = {
        'i2c_address': {
            'boost': [r'(?:i2c|slave)\s+address[\s:=]*(0x[0-9a-fA-F]{2})', 
                     r'device\s+address[\s:=]*(0x[0-9a-fA-F]{2})'],
            'penalize': [r'clock', r'scl.*frequency', r'pin\s+number'],
            'weight': 15.0
        },
        'voltage': {
            'boost': [r'supply\s+voltage', r'operating\s+voltage', r'vcc', r'vdd'],
            'penalize': [r'output\s+voltage', r'reference\s+voltage'],
            'weight': 8.0
        },
        'frequency': {
            'boost': [r'clock\s+frequency', r'max.*frequency', r'operating.*frequency'],
            'penalize': [],
            'weight': 8.0
        },
        'register': {
            'boost': [r'register\s+(?:map|address|table)', r'register.*0x[0-9a-fA-F]{2}'],
            'penalize': [],
            'weight': 10.0
        },
        'pin': {
            'boost': [r'pin\s+(?:configuration|assignment|description)', r'(?:sda|scl|sck).*pin'],
            'penalize': [],
            'weight': 8.0
        }
    }
    
    # Apply pattern-based scoring
    if question_type in patterns:
        pattern_set = patterns[question_type]
        
        # Boost patterns
        for pattern in pattern_set['boost']:
            if re.search(pattern, s_lower):
                score += pattern_set['weight']
        
        # Penalize patterns
        for pattern in pattern_set['penalize']:
            if re.search(pattern, s_lower):
                score -= pattern_set['weight'] * 0.6
    
    # Word overlap (improved)
    q_words = set(w for w in q_lower.split() if len(w) > 2)
    s_words = set(w for w in s_lower.split() if len(w) > 2)
    overlap = len(q_words & s_words)
    score += overlap * 1.2
    
    # Extract and score entities
    entities = extract_entities(sentence)
    entity_count = sum(len(v) for v in entities.values())
    score += entity_count * 0.8
    
    # Technical density
    tech_indicators = ['register', 'address', 'voltage', 'current', 'frequency', 
                       'pin', 'interface', 'protocol', 'configuration', 'control',
                       'data', 'status', 'mode', 'enable', 'disable']
    tech_count = sum(1 for word in tech_indicators if word in s_lower)
    score += tech_count * 0.6
    
    # Contains numbers with context
    if re.search(r'\b\d+(?:\.\d+)?\s*(?:[a-zA-Z]+|[°%])', sentence):
        score += 1.5
    
    # Sentence quality
    if 20 < len(sentence) < 150:  # Prefer moderate length
        score += 1.0
    elif len(sentence) > 200:
        score -= 0.5
    
    # Avoid generic sentences
    generic_phrases = ['see', 'refer to', 'for more information', 'contact', 'note that']
    if any(phrase in s_lower for phrase in generic_phrases):
        score -= 2.0
    
    return score


def determine_question_type(question):
    """Classify question type for targeted extraction"""
    q_lower = question.lower()
    
    if any(term in q_lower for term in ['i2c address', 'slave address', 'device address']):
        return 'i2c_address'
    elif any(term in q_lower for term in ['voltage', 'supply', 'vcc', 'vdd', 'power']):
        return 'voltage'
    elif any(term in q_lower for term in ['frequency', 'clock', 'speed', 'rate']):
        return 'frequency'
    elif any(term in q_lower for term in ['register', 'reg', 'memory map']):
        return 'register'
    elif any(term in q_lower for term in ['pin', 'pinout', 'connection', 'scl', 'sda']):
        return 'pin'
    else:
        return 'general'


def best_sentence_extraction(text, question):
    """Improved sentence extraction with question type awareness"""
    if not text:
        return ""
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?:])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 15]
    
    if not sentences:
        return text[:200]
    
    # Determine question type
    question_type = determine_question_type(question)
    
    # Score all sentences
    scored_sentences = []
    for sent in sentences:
        score = score_sentence_advanced(sent, question, question_type)
        scored_sentences.append((score, sent))
    
    # Sort by score
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    # Return best sentence
    return scored_sentences[0][1] if scored_sentences else text[:200]


def smart_trim(text, max_chars=MAX_ANSWER_CHARS):
    """Improved trimming that preserves meaning"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    
    # Try to cut at sentence boundary
    cut = text[:max_chars]
    sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    
    if sentence_end >= int(max_chars * 0.5):
        return cut[:sentence_end + 1].strip()
    
    # Try to cut at clause boundary
    clause_end = max(cut.rfind(","), cut.rfind(";"), cut.rfind(":"))
    if clause_end >= int(max_chars * 0.6):
        return cut[:clause_end + 1].strip()
    
    # Cut at word boundary
    last_space = cut.rfind(" ")
    if last_space > int(max_chars * 0.7):
        return cut[:last_space] + "…"
    
    return cut + "…"


def hybrid_retrieve_and_answer(question, faiss_index, bm25_index, chunks, model, thresh=CONFIDENCE_THRESHOLD):
    """Enhanced hybrid retrieval with better ranking"""
    
    # FAISS semantic search
    q_emb = model.encode([question], convert_to_numpy=True)
    q_emb = q_emb.astype('float32')
    faiss.normalize_L2(q_emb)
    faiss_scores, faiss_indices = faiss_index.search(q_emb, RETRIEVAL_TOP_K)
    
    # Prepare FAISS results with scores
    faiss_results = [(int(idx), float(score)) for idx, score in zip(faiss_indices[0], faiss_scores[0])]
    
    # BM25 keyword search
    q_tokens = question.lower().split()
    bm25_scores = bm25_index.get_scores(q_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:RETRIEVAL_TOP_K]
    bm25_results = [(int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices]
    
    # Advanced RRF fusion
    fused = advanced_rrf_fusion(faiss_results, bm25_results)
    
    # Collect results
    results = []
    for idx, rrf_score in fused[:7]:
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
    
    # Check confidence
    if not results or results[0]["score"] < thresh:
        return {"answer": "• Answer not found in datasheet.", "results": results}
    
    # Multi-chunk answer extraction for better coverage
    candidate_answers = []
    for result in results[:3]:  # Check top 3 chunks
        extracted = best_sentence_extraction(result["text"], question)
        if extracted:
            candidate_answers.append({
                "text": extracted,
                "score": result["score"],
                "page": result["page"]
            })
    
    # Select best answer
    if candidate_answers:
        best_answer = max(candidate_answers, key=lambda x: x["score"])
        best_sent = smart_trim(best_answer["text"], MAX_ANSWER_CHARS)
        page_ref = f" (pg {best_answer['page']})" if best_answer.get('page') else ""
        answer = f"• {best_sent}{page_ref}"
    else:
        top = results[0]
        best_sent = best_sentence_extraction(top["text"], question)
        best_sent = smart_trim(best_sent, MAX_ANSWER_CHARS)
        page_ref = f" (pg {top['page']})" if top.get('page') else ""
        answer = f"• {best_sent}{page_ref}"
    
    return {"answer": answer, "results": results}


def extract_i2c_info(chunks):
    """Enhanced I2C info extraction with better filtering"""
    i2c_addresses = []
    registers = {}
    voltage_info = []
    pin_info = []
    
    # Refined patterns
    addr_patterns = [
        (r'(?:I2C|i2c)\s+(?:slave\s+)?address[\s:=]*(0x[0-9a-fA-F]{2})', 20),  # Highest priority
        (r'(?:device|slave)\s+address[\s:=]*(0x[0-9a-fA-F]{2})', 15),
        (r'address[\s:=]*(0x[0-9a-fA-F]{2})', 5),  # Lowest priority
    ]
    
    for chunk in chunks:
        text = chunk["text"]
        
        # Extract I2C addresses with priority scoring
        for pattern, priority in addr_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for addr in matches:
                # Context filtering
                context = text[max(0, text.find(addr)-80):text.find(addr)+80].lower()
                
                # Skip if it's clearly not an I2C address
                if any(skip in context for skip in ['clock speed', 'frequency', 'scl rate', 'pin number']):
                    continue
                
                # Add with priority weight
                i2c_addresses.extend([addr] * priority)
        
        # Registers
        reg_matches = re.findall(r'(?:register|reg)[\s_]+([\w]+)[\s:=]+(0x[0-9a-fA-F]{2,4})', text, re.IGNORECASE)
        for reg_name, reg_addr in reg_matches:
            registers[reg_name.upper()] = reg_addr
        
        # Voltage (more specific)
        volt_matches = re.findall(r'(?:supply|operating|vcc|vdd)[\s\w]*voltage[\s:=]*(\d+\.?\d*)\s*(?:v|volt)', text, re.IGNORECASE)
        voltage_info.extend(volt_matches)
        
        # Pins
        pin_matches = re.findall(r'(?:SCL|SDA)[\s:=]+(?:pin\s+)?(\w+)', text, re.IGNORECASE)
        pin_info.extend(pin_matches)
    
    # Select most common I2C address
    i2c_address = "0x00"
    if i2c_addresses:
        i2c_address = Counter(i2c_addresses).most_common(1)[0][0]
    
    # Select most common voltage
    voltage = "3.3"
    if voltage_info:
        voltage = Counter(voltage_info).most_common(1)[0][0]
    
    return {
        "i2c_address": i2c_address,
        "registers": registers,
        "voltage": voltage,
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
    st.markdown("### 💡 Improvements")
    st.caption("✓ Sentence-aware chunking")
    st.caption("✓ Question-type classification")
    st.caption("✓ Weighted RRF fusion")
    st.caption("✓ Entity extraction")
    st.caption("✓ Multi-chunk candidate ranking")

uploaded = st.file_uploader(
    "📤 Upload Datasheet PDF",
    type=["pdf"],
    help="Upload a sensor or MCU datasheet"
)

if uploaded:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded.name:
        with st.spinner("🔄 Processing PDF with enhanced indexing..."):
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
        
        st.success(f"✅ Indexed {len(chunks)} chunks from {len(pages)} pages")
    
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
                with st.spinner("Searching with enhanced retrieval..."):
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
                    st.caption(f"Confidence: {result['results'][0]['score']:.2f} | RRF: {result['results'][0]['rrf_score']:.3f}")
                    
                    with st.expander("📚 View source chunks"):
                        for i, res in enumerate(result["results"][:5], 1):
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
        
        if st.button("🔧 Generate Arduino Code", type="primary", use_container_width=True):
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
    st.markdown("### Enhanced Features:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔍 Smart Chunking")
        st.caption("Sentence-aware boundaries preserve context")
    
    with col2:
        st.markdown("#### 🎯 Question Types")
        st.caption("Specialized scoring for address/voltage/pin queries")
    
    with col3:
        st.markdown("#### ⚡ Better Fusion")
        st.caption("Weighted RRF with semantic + keyword bonus")

st.divider()
st.caption("Built with Streamlit • Enhanced Non-LLM Retrieval")