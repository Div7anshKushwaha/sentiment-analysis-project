# 🎬 Movie Review Sentiment Analyzer

An instant sentiment analysis app powered by a **pretrained DistilBERT model** — no training required, just paste a review and get results.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ✨ Features

- **Single review** analysis with confidence score
- **Bulk analysis** — paste multiple reviews, get a results table
- **Quick examples** to try instantly
- Zero training needed — model loads once and is cached

## 🧠 Model

`distilbert-base-uncased-finetuned-sst-2-english`

- Fine-tuned on **Stanford Sentiment Treebank (SST-2)** — movie review dataset
- ~95% accuracy on sentiment classification
- Model size: ~260MB (auto-downloaded on first run from HuggingFace Hub)

## 🛠 Tech Stack

Python · HuggingFace Transformers · DistilBERT · Streamlit · Pandas · Matplotlib
