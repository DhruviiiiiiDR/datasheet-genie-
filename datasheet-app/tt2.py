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
    """Extract text from PDF with error handling"""
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
    """Improved chunking with sentence boundaries"""
    chunks = []
    
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        
        sentences = re.split(r'(?<=[.!?:])\s+|\n{2,}', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) > 30:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "chunk_id": len(chunks)
                    })
                
                overlap_words = ' '.join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_length += sentence_length
        
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
    
    for rank, (idx, score) in enumerate(faiss_results):
        rrf_score = faiss_weight * (1 / (k + rank + 1))
        semantic_bonus = faiss_weight * (score * 0.3)
        scores[idx] = scores.get(idx, 0) + rrf_score + semantic_bonus
    
    for rank, (idx, score) in enumerate(bm25_results):
        rrf_score = bm25_weight * (1 / (k + rank + 1))
        keyword_bonus = bm25_weight * (min(score / 10, 1) * 0.2)
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
    
    if question_type in patterns:
        pattern_set = patterns[question_type]
        for pattern in pattern_set['boost']:
            if re.search(pattern, s_lower):
                score += pattern_set['weight']
        for pattern in pattern_set['penalize']:
            if re.search(pattern, s_lower):
                score -= pattern_set['weight'] * 0.6
    
    q_words = set(w for w in q_lower.split() if len(w) > 2)
    s_words = set(w for w in s_lower.split() if len(w) > 2)
    overlap = len(q_words & s_words)
    score += overlap * 1.2
    
    entities = extract_entities(sentence)
    entity_count = sum(len(v) for v in entities.values())
    score += entity_count * 0.8
    
    tech_indicators = ['register', 'address', 'voltage', 'current', 'frequency', 
                       'pin', 'interface', 'protocol', 'configuration', 'control',
                       'data', 'status', 'mode', 'enable', 'disable']
    tech_count = sum(1 for word in tech_indicators if word in s_lower)
    score += tech_count * 0.6
    
    if re.search(r'\b\d+(?:\.\d+)?\s*(?:[a-zA-Z]+|[°%])', sentence):
        score += 1.5
    
    if 20 < len(sentence) < 150:
        score += 1.0
    elif len(sentence) > 200:
        score -= 0.5
    
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
    
    sentences = re.split(r'(?<=[.!?:])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 15]
    
    if not sentences:
        return text[:200]
    
    question_type = determine_question_type(question)
    
    scored_sentences = []
    for sent in sentences:
        score = score_sentence_advanced(sent, question, question_type)
        scored_sentences.append((score, sent))
    
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    return scored_sentences[0][1] if scored_sentences else text[:200]


def smart_trim(text, max_chars=MAX_ANSWER_CHARS):
    """Improved trimming that preserves meaning"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    
    cut = text[:max_chars]
    sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    
    if sentence_end >= int(max_chars * 0.5):
        return cut[:sentence_end + 1].strip()
    
    clause_end = max(cut.rfind(","), cut.rfind(";"), cut.rfind(":"))
    if clause_end >= int(max_chars * 0.6):
        return cut[:clause_end + 1].strip()
    
    last_space = cut.rfind(" ")
    if last_space > int(max_chars * 0.7):
        return cut[:last_space] + "…"
    
    return cut + "…"


def hybrid_retrieve_and_answer(question, faiss_index, bm25_index, chunks, model, thresh=CONFIDENCE_THRESHOLD):
    """Enhanced hybrid retrieval with better ranking"""
    
    q_emb = model.encode([question], convert_to_numpy=True)
    q_emb = q_emb.astype('float32')
    faiss.normalize_L2(q_emb)
    faiss_scores, faiss_indices = faiss_index.search(q_emb, RETRIEVAL_TOP_K)
    
    faiss_results = [(int(idx), float(score)) for idx, score in zip(faiss_indices[0], faiss_scores[0])]
    
    q_tokens = question.lower().split()
    bm25_scores = bm25_index.get_scores(q_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:RETRIEVAL_TOP_K]
    bm25_results = [(int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices]
    
    fused = advanced_rrf_fusion(faiss_results, bm25_results)
    
    results = []
    for idx, rrf_score in fused[:7]:
        if idx < 0 or idx >= len(chunks):
            continue
        
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
        return {"answer": "• Answer not found in datasheet.", "results": results, "entities": {}}
    
    candidate_answers = []
    for result in results[:3]:
        extracted = best_sentence_extraction(result["text"], question)
        if extracted:
            candidate_answers.append({
                "text": extracted,
                "score": result["score"],
                "page": result["page"]
            })
    
    if candidate_answers:
        best_answer = max(candidate_answers, key=lambda x: x["score"])
        best_sent = smart_trim(best_answer["text"], MAX_ANSWER_CHARS)
        page_ref = f" (pg {best_answer['page']})" if best_answer.get('page') else ""
        answer = f"• {best_sent}{page_ref}"
        entities = extract_entities(best_sent)
    else:
        top = results[0]
        best_sent = best_sentence_extraction(top["text"], question)
        best_sent = smart_trim(best_sent, MAX_ANSWER_CHARS)
        page_ref = f" (pg {top['page']})" if top.get('page') else ""
        answer = f"• {best_sent}{page_ref}"
        entities = extract_entities(best_sent)
    
    return {"answer": answer, "results": results, "entities": entities}


@st.cache_data
def extract_i2c_info(_chunks):
    """ENHANCED I2C info extraction with better patterns"""
    i2c_addresses = []
    registers = {}
    voltage_info = []
    pin_info = []
    
    # COMPREHENSIVE I2C ADDRESS PATTERNS
    addr_patterns = [
        (r'(?:I2C|i2c)\s+(?:slave\s+)?address[\s:=]*(0x[0-9a-fA-F]{2})', 25),
        (r'(?:device|slave)\s+address[\s:=]*(0x[0-9a-fA-F]{2})', 20),
        (r'(?:7-bit|7bit)\s+address[\s:=]*(0x[0-9a-fA-F]{2})', 18),
        (r'address[\s:]*(?:is|=|:)\s*(0x[0-9a-fA-F]{2})', 15),
        (r'address.*?(0x[0-9a-fA-F]{2})', 10),
        (r'(0x[0-9a-fA-F]{2}).*?address', 10),
        (r'slave\s+addr[\s:=]*(0x[0-9a-fA-F]{2})', 15),
        (r'chip\s+address[\s:=]*(0x[0-9a-fA-F]{2})', 15),
    ]
    
    # ENHANCED REGISTER PATTERNS
    reg_patterns = [
        r'(?:register|reg)[\s_]+([\w]+)[\s:=]+(0x[0-9a-fA-F]{2,4})',
        r'(0x[0-9a-fA-F]{2,4})[\s:]+(\w+)[\s_]*(?:register|reg)',
        r'(?:REG|Reg)_([\w]+)[\s:=]+(0x[0-9a-fA-F]{2,4})',
        r'([\w]+)_(?:REG|reg)[\s:=]+(0x[0-9a-fA-F]{2,4})',
        r'(?:CONFIG|CTRL|DATA|STATUS)[\s:=]+(0x[0-9a-fA-F]{2,4})',
    ]
    
    for chunk in _chunks:
        text = chunk["text"]
        
        # Extract I2C addresses with priority
        for pattern, priority in addr_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for addr in matches:
                context = text[max(0, text.find(addr)-100):text.find(addr)+100].lower()
                
                skip_terms = ['clock speed', 'frequency', 'scl rate', 'pin number', 'baud', 'speed']
                if any(skip in context for skip in skip_terms):
                    continue
                
                if 'i2c' in context or 'slave' in context:
                    priority += 10
                
                i2c_addresses.extend([addr] * priority)
        
        # Extract registers
        for reg_pattern in reg_patterns:
            reg_matches = re.findall(reg_pattern, text, re.IGNORECASE)
            for match in reg_matches:
                if isinstance(match, tuple):
                    if match[0].startswith('0x'):
                        reg_name = match[1].upper().replace('_REG', '').replace('REG_', '')
                        reg_addr = match[0]
                    else:
                        reg_name = match[0].upper().replace('_REG', '').replace('REG_', '')
                        reg_addr = match[1]
                    registers[reg_name] = reg_addr
                else:
                    registers[match.upper()] = "0x00"
        
        # Voltage
        volt_matches = re.findall(
            r'(?:supply|operating|vcc|vdd|input)[\s\w]*voltage[\s:=]*(\d+\.?\d*)\s*(?:v|volt)',
            text, re.IGNORECASE
        )
        voltage_info.extend(volt_matches)
        
        # Pins
        pin_matches = re.findall(r'(?:SCL|SDA)[\s:=]+(?:pin\s+)?([A-Z]?\d+|\w+)', text, re.IGNORECASE)
        pin_info.extend(pin_matches)
    
    # Select most common I2C address (filter invalid)
    i2c_address = "0x00"
    if i2c_addresses:
        counter = Counter(i2c_addresses)
        valid_addresses = {addr: count for addr, count in counter.items() 
                          if addr.lower() not in ['0x00', '0xff']}
        if valid_addresses:
            i2c_address = max(valid_addresses, key=valid_addresses.get)
        elif i2c_addresses:
            i2c_address = counter.most_common(1)[0][0]
    
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
    st.markdown("### 💡 Features")
    st.caption("✓ Smart question answering")
    st.caption("✓ Auto I2C detection")
    st.caption("✓ Manual value override")
    st.caption("✓ Arduino code generation")
    st.caption("✓ Error handling")


uploaded = st.file_uploader(
    "📤 Upload Datasheet PDF",
    type=["pdf"],
    help="Upload a sensor or MCU datasheet"
)


if uploaded:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded.name:
        with st.spinner("🔄 Processing PDF..."):
            try:
                pages = extract_pdf_text(uploaded)
                if not pages:
                    st.error("Failed to extract text from PDF.")
                    st.stop()
                
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
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("💬 Ask Questions")
        
        with st.expander("💡 Example Questions"):
            example_questions = [
                "What is the I2C address?",
                "What is the operating voltage?",
                "What are the configuration registers?",
                "Which pins are SCL and SDA?",
                "What is the power consumption?"
            ]
            for eq in example_questions:
                if st.button(eq, key=f"ex_{eq}"):
                    st.session_state["current_question"] = eq
        
        question = st.text_input(
            "Your question:",
            value=st.session_state.get("current_question", ""),
            placeholder="e.g., What is the I2C address?",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Ask", type="primary", use_container_width=True):
            if question:
                with st.spinner("Searching..."):
                    try:
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
                            
                            if result["results"]:
                                top_score = result["results"][0]["score"]
                                st.info(f"💡 Top score ({top_score:.2f}) below threshold ({confidence:.2f}). Try lowering the slider.")
                            else:
                                st.info("💡 No relevant chunks found. Try rephrasing.")
                        else:
                            st.success(result["answer"])
                            
                            entities = result.get("entities", {})
                            entity_summary = [f"{k}: {len(v)}" for k, v in entities.items() if v]
                            if entity_summary:
                                st.caption(f"🔍 Detected: {', '.join(entity_summary)}")
                        
                        if result["results"]:
                            st.caption(f"Confidence: {result['results'][0]['score']:.2f} | RRF: {result['results'][0]['rrf_score']:.3f}")
                            
                            with st.expander("📚 View source chunks"):
                                for i, res in enumerate(result["results"][:5], 1):
                                    st.markdown(f"**Chunk {i}** (Page {res['page']}, Score: {res['score']:.2f})")
                                    st.text(res['text'][:400] + "…")
                                    st.divider()
                    except Exception as e:
                        st.error(f"Error during search: {e}")
    
    with col_right:
        st.subheader("⚙️ Arduino I2C Code")
        
        i2c_info = st.session_state["i2c_info"]
        
        # MANUAL OVERRIDE
        with st.expander("✏️ Edit Detected Values"):
            manual_i2c = st.text_input("I2C Address:", value=i2c_info['i2c_address'], key="manual_i2c")
            manual_voltage = st.text_input("Voltage (V):", value=str(i2c_info['voltage']), key="manual_voltage")
            
            if st.button("💾 Update Values"):
                i2c_info['i2c_address'] = manual_i2c
                i2c_info['voltage'] = manual_voltage
                st.session_state["i2c_info"] = i2c_info
                st.success("✅ Values updated!")
                st.rerun()
        
        st.markdown("**Detected Info:**")
        st.code(f"I2C Address: {i2c_info['i2c_address']}\nRegisters: {len(i2c_info['registers'])}\nVoltage: {i2c_info['voltage']}V", language="text")
        
        device_name = st.text_input(
            "Device name:",
            value="Sensor",
            help="Used in code comments"
        )
        
        if st.button("🔧 Generate Arduino Code", type="primary", use_container_width=True):
            with st.spinner("Generating code..."):
                try:
                    # WARNING if using defaults
                    if i2c_info['i2c_address'] == "0x00":
                        st.warning("⚠️ I2C address not detected. Using default 0x00. Please edit manually above.")
                    
                    arduino_code = generate_arduino_i2c_code(i2c_info, device_name=device_name)
                    st.session_state["generated_code"] = arduino_code
                    st.success("✅ Code generated!")
                except Exception as e:
                    st.error(f"Error generating code: {e}")
        
        # DISPLAY CODE
        if "generated_code" in st.session_state:
            arduino_code = st.session_state["generated_code"]
            
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
    st.markdown("### Why Use This App?")
    
    st.markdown("""
    - 🔍 **Find specs instantly** - No more scrolling through 100+ page PDFs
    - 💬 **Ask natural questions** - "What's the I2C address?" instead of searching manually  
    - ⚡ **Save hours** - Auto-generates working Arduino code from the datasheet
    - 📋 **Ready to use** - Download code and paste into Arduino IDE
    """)


st.divider()
st.caption("Built with Streamlit • Enhanced Non-LLM RAG")
