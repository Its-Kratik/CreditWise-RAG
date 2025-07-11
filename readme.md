# 🤖 CreditWise-RAG

**Smart RAG Q&A Chatbot for Loan Prediction Datasets**

Built by [Kratik Jain](mailto:kratikjain121@gmail.com)

---

### 💡 What is CreditWise-RAG?
A powerful Retrieval-Augmented Generation (RAG) system designed to:

- ✅ Answer **math-based queries** from CSV datasets using a smart dispatcher
- 🤖 Fallback to **Gemini 1.5 Flash LLM** when logic isn't enough
- 🧠 Auto-classify query types using **regex + spaCy NLP**
- 🖼️ Offer both CLI and Streamlit UI for interaction

---

### 📁 Project Structure
```
CreditWise-RAG/
├── rag_math_engine/         # Core logic engine
│   ├── core.py              # Math handlers + dispatcher
│   ├── patterns.py          # Regex + spaCy classifier
│   └── __init__.py
│
├── llm_fallback.py         # FAISS + Gemini LLM answer generator
├── app.py                  # Streamlit UI
├── main.py                 # CLI interface
├── requirements.txt
├── setup.py
└── tests/
    └── test_handlers.py    # Unit tests
```

---

### 🚀 Quickstart

#### 1. 📦 Install dependencies
```bash
pip install -e .
python -m spacy download en_core_web_sm
```

#### 2. 🧪 Run tests
```bash
pytest tests/
```

#### 3. 🖼️ Launch the Streamlit app
```bash
streamlit run app.py
```

#### 4. 🧭 Use CLI
```bash
python main.py --csv path/to/data.csv --mode math --question "average loan for graduates"
```

---

### 🔍 Example Questions Supported
- What is the average loan amount for graduates?
- How many loans were approved?
- What is the approval rate for self-employed?
- Median loan amount?
- Income-wise or gender-wise stats?

---

### 🤖 Gemini LLM Fallback
Make sure you set your API key:
```bash
export GOOGLE_API_KEY="your-key-here"
```

LLM kicks in when math engine can't classify the query.

---

### 📧 Contact
Made with 💡 by **Kratik Jain**
- GitHub: [Its-Kratik](https://github.com/Its-Kratik)
- Email: kratikjain121@gmail.com

---

### 🌟 Star this repo if you find it useful!
