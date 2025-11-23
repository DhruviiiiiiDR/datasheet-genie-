import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import ollama
import time

st.title("📄 Fast Datasheet RAG with Ollama")

# --- Utility functions ---

def extract_pdf_text(file):
    start = time.time()
    file.seek(0)
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in pdf:
        full_text += page.get_text() + "\n"
    pdf.close()
    st.write(f"⏱️ Text extraction took {time.time()-start:.2f}s")
    return full_text

def chunk_text(text, chunk_size=300):
    start = time.time()
    words = text.split()
    st.write(f"📊 Total words in document: {len(words)}")
    # Limit to first 5000 words for even more speed
    words = words[:5000]
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    st.write(f"⏱️ Chunking took {time.time()-start:.2f}s")
    return chunks

def get_ollama_embedding(text):
    """Use Ollama's embedding model"""
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return np.array(response["embedding"])
    except Exception as e:
        st.error(f"Embedding error: {e}")
        st.info("Run: `ollama pull nomic-embed-text`")
        raise

def compute_embeddings_and_index(chunks):
    """Compute embeddings using Ollama"""
    start = time.time()
    st.info(f"🔄 Computing embeddings for {len(chunks)} chunks...")
    embeddings = []
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    for i, chunk in enumerate(chunks):
        chunk_start = time.time()
        emb = get_ollama_embedding(chunk)
        embeddings.append(emb)
        progress_bar.progress((i + 1) / len(chunks))
        
        # Show timing every 5 chunks
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(chunks) - i - 1)
            status.write(f"Chunk {i+1}/{len(chunks)} | Avg: {avg_time:.2f}s/chunk | Est. remaining: {remaining:.1f}s")
    
    st.write(f"⏱️ Embedding computation took {time.time()-start:.2f}s")
    
    # Build FAISS index
    index_start = time.time()
    embeddings = np.array(embeddings).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    st.write(f"⏱️ FAISS index built in {time.time()-index_start:.2f}s")
    
    return embeddings, index

def retrieve_relevant_chunks(question, chunks, faiss_index):
    """Simple FAISS-only retrieval"""
    start = time.time()
    q_emb = get_ollama_embedding(question)
    st.write(f"⏱️ Question embedding: {time.time()-start:.2f}s")
    
    q_emb = np.array([q_emb]).astype('float32')
    faiss.normalize_L2(q_emb)
    
    search_start = time.time()
    _, faiss_res = faiss_index.search(q_emb, 5)
    st.write(f"⏱️ FAISS search: {time.time()-search_start:.2f}s")
    
    return [chunks[i] for i in faiss_res[0]]

# Test Ollama connectivity first
st.sidebar.subheader("🔍 System Check")
try:
    test_start = time.time()
    test_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hi"}],
        options={"num_predict": 5}
    )
    test_time = time.time() - test_start
    st.sidebar.success(f"✅ Ollama responding ({test_time:.2f}s)")
except Exception as e:
    st.sidebar.error(f"❌ Ollama not responding: {e}")
    st.error("Ollama is not working. Make sure it's running: `ollama serve`")

# Check if embedding model exists
try:
    embed_start = time.time()
    test_emb = ollama.embeddings(model="nomic-embed-text", prompt="test")
    embed_time = time.time() - embed_start
    st.sidebar.success(f"✅ Embeddings working ({embed_time:.2f}s)")
except Exception as e:
    st.sidebar.error("❌ nomic-embed-text not found")
    st.sidebar.code("ollama pull nomic-embed-text")

# --- App flow ---

uploaded = st.file_uploader("Upload your datasheet PDF", type=["pdf"])

if uploaded:
    st.success("✅ File uploaded!")

    # Step 1: Extract text
    if "raw_text" not in st.session_state:
        with st.spinner("Extracting text from PDF..."):
            st.session_state['raw_text'] = extract_pdf_text(uploaded)
        st.success("✅ Text extracted!")

    st.subheader("📄 Datasheet Preview")
    preview_text = st.session_state['raw_text'][:800]
    st.text_area("Preview", preview_text, height=150)

    # Step 2: Chunk text
    if "chunks" not in st.session_state:
        with st.spinner("Chunking text..."):
            st.session_state['chunks'] = chunk_text(st.session_state['raw_text'])
        st.write(f"✅ Created {len(st.session_state['chunks'])} chunks")

    # Step 3: Build index
    if "faiss_index" not in st.session_state:
        st.warning("⚠️ Building index - this will take time on first run...")
        try:
            embeddings, faiss_index = compute_embeddings_and_index(st.session_state['chunks'])
            st.session_state['embeddings'] = embeddings
            st.session_state['faiss_index'] = faiss_index
            st.success("✅ Index ready!")
        except Exception as e:
            st.error(f"Error building index: {e}")
            st.stop()
    else:
        st.success("✅ Using cached index (fast!)")

    # Step 4: Question Answering
    st.subheader("❓ Ask Questions")
    question = st.text_input("Your question:")

    if question and st.button("🚀 Get Answer"):
        total_start = time.time()
        try:
            with st.spinner("Finding relevant information..."):
                relevant_chunks = retrieve_relevant_chunks(
                    question, 
                    st.session_state['chunks'],
                    st.session_state['faiss_index']
                )
            
            st.write(f"✅ Found {len(relevant_chunks)} relevant sections")
            
            # Show context
            with st.expander("📋 Retrieved Context"):
                for i, chunk in enumerate(relevant_chunks):
                    st.text_area(f"Section {i+1}", chunk, height=100, key=f"chunk_{i}")

            prompt = f"""Based on this datasheet information, answer the question concisely.

Context:
{chr(10).join(relevant_chunks)}

Question: {question}

Answer:"""
            
            llm_start = time.time()
            with st.spinner("Generating answer..."):
                response = ollama.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1, "num_predict": 150}
                )
            st.write(f"⏱️ LLM generation: {time.time()-llm_start:.2f}s")
            
            st.subheader("✨ Answer")
            st.write(response["message"]["content"])
            
            st.info(f"⏱️ Total time: {time.time()-total_start:.2f}s")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Step 5: I2C Code Generation
    st.subheader("⚙️ Generate Arduino I2C Code")

    if st.button("Generate I2C Code"):
        code_start = time.time()
        try:
            # Use first 2000 chars only for speed
            datasheet_excerpt = st.session_state['raw_text'][:2000]
            
            prompt = f"""Analyze this sensor datasheet and generate Arduino I2C code.

Datasheet:
{datasheet_excerpt}

Generate complete Arduino code with:
1. I2C address and registers
2. Wire.begin() initialization
3. Functions to read sensor data
4. Example in loop()

Provide only the code:"""

            with st.spinner("Generating code..."):
                code_response = ollama.chat(
                    model='llama3.2',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={"temperature": 0.2, "num_predict": 400}
                )

            generated_code = code_response['message']['content']
            st.code(generated_code, language='cpp')
            
            st.info(f"⏱️ Code generation took: {time.time()-code_start:.2f}s")
            
            st.download_button(
                label="⬇️ Download Arduino Code",
                data=generated_code,
                file_name="sensor_driver.ino",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.sidebar.markdown("""
### ⚡ Speed Tips:
- First run is slow (building index)
- Subsequent questions are fast!
- Reduced to 5000 words max
- Using 300-word chunks

### 📋 Required:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```
""")