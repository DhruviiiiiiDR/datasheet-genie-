import streamlit as st
import fitz  # PyMuPDF
import ollama
from rank_bm25 import BM25Okapi
import re

st.set_page_config(page_title="Datasheet RAG", page_icon="📄", layout="wide")
st.title("📄 Datasheet RAG System")

# --- Functions ---

def extract_pdf_text(file):
    """Extract text from PDF"""
    file.seek(0)
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text() + "\n"
    pdf.close()
    return text

def create_chunks(text, chunk_size=400, overlap=100):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:
            chunks.append(chunk)
    return chunks

def retrieve_chunks(query, chunks, bm25, top_k=5):
    """Retrieve most relevant chunks using BM25"""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]

def generate_answer(question, context, model="llama3.2:latest"):
    """Generate answer using Ollama"""
    prompt = f"""You are a helpful assistant analyzing a technical datasheet. Answer the question based ONLY on the provided context.

Context from datasheet:
{context}

Question: {question}

Provide a clear, direct answer. If the information is not in the context, say so.

Answer:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 200}
    )
    return response["message"]["content"]

def generate_i2c_code(datasheet_text, model="llama3.2:latest"):
    """Generate Arduino I2C code from datasheet"""
    # Find I2C related sections
    i2c_sections = []
    lines = datasheet_text.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'i2c|I2C|address|register', line, re.IGNORECASE):
            context = ' '.join(lines[max(0, i-2):min(len(lines), i+3)])
            i2c_sections.append(context)
    
    i2c_context = '\n'.join(i2c_sections[:10])  # Limit context
    
    prompt = f"""Based on this sensor datasheet information, generate a complete Arduino I2C driver.

Datasheet I2C Information:
{i2c_context[:2000]}

Generate a complete Arduino sketch (.ino file) that includes:
1. Wire library initialization
2. I2C device address as #define
3. Key register addresses as #define
4. setup() function with Wire.begin() and sensor initialization
5. loop() function that reads sensor data
6. Serial.print() statements to display data

Only output the Arduino code, no explanations:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_predict": 600}
    )
    
    code = response["message"]["content"]
    # Clean markdown if present
    code = re.sub(r'```.*?\n', '', code)
    code = re.sub(r'```', '', code)
    return code.strip()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model:",
        ["llama3.2:latest", "qwen2.5:3b", "phi3:mini"],
        index=0
    )
    st.session_state["selected_model"] = model_choice
    
    st.divider()
    
    # Test Ollama
    try:
        test = ollama.chat(model=model_choice, messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})
        st.success(f"✅ {model_choice} ready")
    except Exception as e:
        st.error(f"❌ Error with {model_choice}")
        st.caption(str(e))
    
    st.divider()
    st.caption("📊 Your models:")
    st.code("• llama3.2:latest (2.0 GB)\n• qwen2.5:3b (1.9 GB)\n• phi3:mini (2.2 GB)\n• nomic-embed-text")

# --- Main App ---

# File upload
uploaded_file = st.file_uploader("Upload Datasheet PDF", type=["pdf"])

if uploaded_file:
    
    # Process PDF
    if "processed" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            # Extract text
            text = extract_pdf_text(uploaded_file)
            st.session_state["text"] = text
            
            # Create chunks
            chunks = create_chunks(text)
            st.session_state["chunks"] = chunks
            
            # Build BM25 index
            tokenized_chunks = [chunk.lower().split() for chunk in chunks]
            bm25 = BM25Okapi(tokenized_chunks)
            st.session_state["bm25"] = bm25
            
            st.session_state["processed"] = True
            st.session_state["file_name"] = uploaded_file.name
        
        st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")
    
    # Show preview
    with st.expander("📄 Document Preview"):
        st.text_area("First 1000 characters", st.session_state["text"][:1000], height=200)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Questions")
        
        # Question input
        question = st.text_input("Your question:", key="question_input", placeholder="e.g., What is the I2C address?")
        
        if st.button("🔍 Get Answer", type="primary") and question:
            with st.spinner("Searching and generating answer..."):
                # Get selected model
                model = st.session_state.get("selected_model", "llama3.2:latest")
                
                # Retrieve relevant chunks
                relevant_chunks = retrieve_chunks(
                    question, 
                    st.session_state["chunks"], 
                    st.session_state["bm25"]
                )
                
                # Generate answer
                answer = generate_answer(question, "\n\n".join(relevant_chunks), model)
                
                # Display
                st.markdown("### Answer")
                st.write(answer)
                
                # Show sources
                with st.expander("📚 View Source Chunks"):
                    for i, chunk in enumerate(relevant_chunks[:3]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(chunk[:300] + "...")
                        st.divider()
    
    with col2:
        st.subheader("⚙️ Arduino Code")
        
        if st.button("🔧 Generate I2C Driver", type="secondary"):
            with st.spinner("Generating Arduino code..."):
                try:
                    model = st.session_state.get("selected_model", "llama3.2:latest")
                    code = generate_i2c_code(st.session_state["text"], model)
                    
                    st.code(code, language="cpp")
                    
                    st.download_button(
                        "⬇️ Download .ino file",
                        code,
                        file_name="sensor_driver.ino",
                        mime="text/x-arduino"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Example questions
    st.markdown("---")
    st.markdown("**Example questions:**")
    examples = [
        "What is the I2C address?",
        "What is the operating voltage?",
        "How do I read temperature data?",
        "What registers need to be configured?"
    ]
    
    cols = st.columns(4)
    for i, example in enumerate(examples):
        if cols[i].button(example, key=f"example_{i}"):
            st.session_state.question_input = example
            st.rerun()

else:
    st.info("👆 Upload a PDF datasheet to get started")
    
    st.markdown("""
    ### How it works:
    1. **Upload** your datasheet PDF
    2. **Ask questions** in natural language
    3. **Generate** Arduino I2C driver code
    
    ### Requirements:
    ```bash
    pip install streamlit pymupdf rank-bm25
    ollama pull llama3.2
    ollama serve
    ```
    """)