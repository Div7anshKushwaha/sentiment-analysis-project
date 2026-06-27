# 🎬 IMDB Sentiment Analysis — Naive Bayes

A Streamlit web app for classifying movie reviews as **Positive** or **Negative** using three Naive Bayes classifiers.

## 📁 Project Structure

```
sentiment_project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🔧 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## 🧠 Pipeline (from notebook)

| Step | Operation |
|------|-----------|
| 1 | Remove HTML tags |
| 2 | Lowercase conversion |
| 3 | Remove special characters |
| 4 | Remove stopwords (NLTK) |
| 5 | Porter Stemming |
| 6 | CountVectorizer (2500 features) |
| 7 | Train 3 Naive Bayes models |

## 📊 Models Compared

- **Gaussian NB** — Assumes Gaussian distribution of features
- **Multinomial NB** — Best for word count data ✅ (usually best here)
- **Bernoulli NB** — Binary presence/absence of words

## 📁 Dataset

Upload an IMDB CSV with columns:
- `review` — movie review text
- `sentiment` — `positive` or `negative`

Download from [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Or use the built-in **Demo Data** for a quick trial.

## 🚀 Features

- Train all 3 NB models with one click
- Accuracy comparison bar chart
- Confusion matrix & classification report
- Real-time sentiment prediction on custom input
- Probability scores with visual gauge
- Preprocessed text inspection
