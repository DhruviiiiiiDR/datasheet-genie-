Here's a professional GitHub README for your project that highlights both the technical depth and practical usefulness. Since you're planning to apply for embedded/software internships, this style is much stronger than a simple README.

---

# 📄 Datasheet RAG + Arduino I2C Code Generator

> An AI-powered Retrieval-Augmented Generation (RAG) system that allows users to query PDF datasheets using natural language and automatically generates Arduino I2C driver code.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![BM25](https://img.shields.io/badge/BM25-Retrieval-orange)
![Ollama](https://img.shields.io/badge/Ollama-Gemma2-purple)

---

## 🚀 Overview

Reading a 200-page sensor datasheet just to find the I2C address or register map is tedious and time-consuming.

This project combines **semantic search**, **keyword retrieval**, and **LLM reasoning** to let users simply ask questions like:

> *"What is the I2C address?"*

or

> *"What is the operating voltage?"*

The application instantly retrieves the most relevant datasheet section and can even generate an **Arduino I2C driver** automatically.

---

## ✨ Features

* 📄 Upload any sensor or MCU datasheet (PDF)
* 🔍 Natural language question answering
* 🧠 Hybrid Retrieval (FAISS + BM25)
* 📑 Smart document chunking
* 🤖 Optional AI explanations using Gemma 2 2B (Ollama)
* ⚙ Automatic I2C information extraction
* ✏ Manual editing of detected values
* 💻 Automatic Arduino I2C driver generation
* 📊 Built-in benchmark testing
* 📥 Download generated `.ino` files

---

# 🏗 System Architecture

```text
                 PDF Datasheet
                       │
                       ▼
             Text Extraction (PyMuPDF)
                       │
                       ▼
              Smart Chunk Generation
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Sentence Embeddings              BM25 Index
 (MiniLM-L6-v2)               Keyword Retrieval
        │                             │
        └──────────────┬──────────────┘
                       ▼
            Reciprocal Rank Fusion (RRF)
                       │
                       ▼
          Best Matching Datasheet Chunks
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
 Rule-based Answer          Gemma 2 2B (Optional)
          │                          │
          └────────────┬─────────────┘
                       ▼
              Final Answer + Sources
                       │
                       ▼
          Arduino I2C Driver Generator
```

---

# 🧠 Technologies Used

| Component       | Technology                   |
| --------------- | ---------------------------- |
| Frontend        | Streamlit                    |
| PDF Parsing     | PyMuPDF (fitz)               |
| Embeddings      | Sentence Transformers        |
| Embedding Model | all-MiniLM-L6-v2             |
| Vector Database | FAISS                        |
| Keyword Search  | BM25                         |
| Ranking         | Reciprocal Rank Fusion (RRF) |
| AI Model        | Gemma 2 2B (via Ollama)      |
| Language        | Python                       |

---

# 🔍 Retrieval Pipeline

The application uses a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline.

### Step 1 – PDF Processing

* Extract text using PyMuPDF
* Preserve page numbers
* Remove empty pages

### Step 2 – Smart Chunking

Documents are split into overlapping chunks while respecting sentence boundaries.

Default settings:

* Chunk Size: **300 words**
* Overlap: **100 words**

---

### Step 3 – Semantic Retrieval

Each chunk is converted into embeddings using:

```
all-MiniLM-L6-v2
```

The embeddings are indexed using **FAISS** for efficient similarity search.

---

### Step 4 – Keyword Retrieval

A BM25 index is built over the document to capture exact keyword matches.

---

### Step 5 – Reciprocal Rank Fusion (RRF)

Instead of relying on a single retrieval method, FAISS and BM25 rankings are combined using **Weighted Reciprocal Rank Fusion**, improving retrieval accuracy.

---

### Step 6 – Answer Extraction

The highest-ranked chunks are scored based on:

* Question type
* Technical entities
* Register names
* Numerical values
* I2C-specific patterns

The most relevant sentence is extracted and displayed.

---

### Step 7 – Optional LLM Reasoning

If retrieval confidence falls below a threshold, the system can query **Gemma 2 2B** running locally through Ollama to generate a concise technical answer.

---

# ⚙ Arduino Code Generation

The application automatically extracts:

* I2C Slave Address
* Register Map
* Operating Voltage
* Device Configuration Registers

It then generates:

* Wire library initialization
* Register read/write functions
* Sensor initialization
* Error handling
* Example data acquisition loop

If register extraction fails, Gemma generates the code directly from the datasheet context.

---

# 📊 Benchmarking

The project includes an integrated benchmarking module.

It evaluates:

* Retrieval time
* Confidence score
* Query difficulty
* LLM usage
* Generated answer quality

Benchmark results are exported as JSON for future analysis.

---

# 📂 Project Structure

```text
├── app.py
├── benchmark_results.json
├── requirements.txt
├── README.md
└── sample_datasheets/
```

---

# 🖥 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/datasheet-rag-i2c-generator.git

cd datasheet-rag-i2c-generator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 🤖 Running with AI (Optional)

Install Ollama.

Pull Gemma 2 2B:

```bash
ollama pull gemma2:2b
```

Start Ollama:

```bash
ollama serve
```

The application automatically connects to:

```
http://localhost:11434
```

---

# 📸 Example Workflow

1. Upload a sensor datasheet.
2. Ask:

```
What is the I2C address?
```

3. Receive an answer with page references.

4. Automatically detect:

* I2C address
* Registers
* Voltage

5. Generate an Arduino `.ino` driver.

6. Download the generated code.

---

# 🎯 Future Improvements

* Support SPI peripherals
* STM32 HAL code generation
* ESP-IDF code generation
* Register table extraction using OCR
* Multi-PDF retrieval
* Chat history memory
* Support for additional LLMs (Llama, Mistral, Phi)


# 👩‍💻 Author

**Dhruvika DR**

Electronics & Communication Engineering
Interested in Embedded Systems, FPGA, AI, and Edge Intelligence.

---

⭐ **If you found this project useful, consider giving it a star!**
