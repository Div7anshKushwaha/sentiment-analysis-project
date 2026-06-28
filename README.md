# 🎬 CineScore — Movie Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat&logo=huggingface)
![Model](https://img.shields.io/badge/Model-DistilBERT-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> An AI-powered movie review sentiment analyzer built with DistilBERT — no training required, just paste a review and get instant results.

🔗 **[Live Demo →](https://sentiment-analysis-project-h7uvvzazvyykvosd8v6sfr.streamlit.app)**

---

## ✨ Features

- **Instant Analysis** — No model training needed, powered by pretrained DistilBERT
- **Confidence Score** — Shows how confident the model is with a visual progress bar
- **Bulk Analysis** — Analyze multiple reviews at once with summary stats
- **Quick Examples** — One-click example reviews to test instantly
- **Cinematic UI** — Dark theme with gold accents, built for aesthetics

---

## 🖥️ Demo

| Single Review | Bulk Analysis |
|---|---|
| Paste any review → get Positive/Negative instantly | One review per line → get full stats table |

---

## 🧠 Model

**`distilbert-base-uncased-finetuned-sst-2-english`**

| Property | Detail |
|---|---|
| Base Model | DistilBERT (distilled BERT) |
| Fine-tuned on | Stanford Sentiment Treebank v2 (SST-2) |
| Accuracy | ~95% on movie reviews |
| Model Size | ~260MB |
| Source | [HuggingFace Hub](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Div7anshKushwaha/sentiment-analysis-project.git
cd sentiment-analysis-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> First run downloads the model (~260MB) — takes 2-3 minutes. Cached after that.

---

## 📁 Project Structure

```
sentiment-analysis-project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
|__ IMDB Dataset.csv    # Dataset
|__ sentiment-analysis-using-naive-bayes.ipyby # main notebook
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web app framework |
| HuggingFace Transformers | Pretrained model loading |
| DistilBERT | Sentiment classification |
| Pandas | Bulk results table |
| Matplotlib | Confidence bar chart |

---

## 📦 Requirements

```
streamlit>=1.32.0
transformers>=4.40.0
torch>=2.0.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## 💭 Why I built this

I wanted to move beyond basic ML projects with .pkl files
and understand how production NLP actually works.
This was my first time using HuggingFace transformers
and deploying a deep learning model — learned a lot!

## 👤 Author

**Divyansh Kushwaha**
- GitHub: [@Div7anshKushwaha](https://github.com/Div7anshKushwaha)
- LinkedIn: [linkedin.com/in/divyansh-kushwaha-603616383](https://linkedin.com/in/divyansh-kushwaha-603616383)

---

*Built as part of an NLP portfolio project | IIT Madras BS Data Science*
