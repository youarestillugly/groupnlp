# Evidence-Grounded Research Assistant

An AI-powered research assistant designed to retrieve and answer questions from scientific research papers related to brain tumor MRI interpretation.

## Features
- Hybrid retrieval using BM25 and dense embeddings
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Evidence-grounded answers
- Page-level citations
- Local LLM integration using Qwen 2.5

## Technologies Used
- Python
- Streamlit
- Sentence Transformers
- BM25
- Cross-Encoder
- Llama.cpp
- Qwen 2.5

## Deployment
The application is deployed using Streamlit Community Cloud.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Domain
Artificial Intelligence in Healthcare — Brain Tumor MRI Research.
