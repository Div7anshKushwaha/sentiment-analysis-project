import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis | Naive Bayes",
    page_icon="🎬",
    layout="wide"
)

# ─── NLTK Downloads ───────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    return stopwords.words('english')

STOP_WORDS = download_nltk()
ps = PorterStemmer()

# ─── Text Preprocessing Functions (from notebook) ─────────────────────────────
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def convert_lower(text):
    return text.lower()

def remove_special(text):
    x = ''
    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + ' '
    return x

def remove_stopwords(text):
    return [word for word in text.split() if word not in STOP_WORDS]

def stem_words(tokens):
    return [ps.stem(w) for w in tokens]

def join_back(tokens):
    return " ".join(tokens)

def preprocess(text):
    text = clean_html(text)
    text = convert_lower(text)
    text = remove_special(text)
    tokens = remove_stopwords(text)
    tokens = stem_words(tokens)
    return join_back(tokens)

# ─── Model Training ───────────────────────────────────────────────────────────
MODEL_PATH = "models/"

@st.cache_resource
def train_models(df):
    os.makedirs(MODEL_PATH, exist_ok=True)

    df = df.copy()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0}).astype(int)

    with st.spinner("🧹 Cleaning and preprocessing reviews..."):
        df['review'] = df['review'].apply(preprocess)

    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(df['review']).toarray()
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Multinomial NB": MultinomialNB(),
        "Bernoulli NB":   BernoulliNB(),
        "Gaussian NB":    GaussianNB(),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "model": clf,
            "accuracy": accuracy_score(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }

    # Pick best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])

    return cv, results, best_name

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 Sentiment Analyzer")
st.sidebar.markdown("**Naive Bayes on IMDB Reviews**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Train & Evaluate", "🔍 Predict"]
)

# ─── Session State ────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False
if "cv" not in st.session_state:
    st.session_state.cv = None
if "results" not in st.session_state:
    st.session_state.results = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "chosen_model_name" not in st.session_state:
    st.session_state.chosen_model_name = "Multinomial NB"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🎬 IMDB Sentiment Analysis")
    st.subheader("Using Naive Bayes Classifiers")

    st.markdown("""
    This project classifies IMDB movie reviews as **Positive** or **Negative** using three Naive Bayes variants.

    ### 📋 Pipeline (from notebook)
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1: Text Cleaning**\n\n- Remove HTML tags\n- Lowercase conversion\n- Remove special characters")
    with col2:
        st.info("**Step 2: NLP Processing**\n\n- Remove stopwords\n- Porter Stemming\n- Bag of Words (2500 features)")
    with col3:
        st.info("**Step 3: Classification**\n\n- Gaussian NB\n- Multinomial NB\n- Bernoulli NB")

    st.divider()
    st.markdown("""
    ### 🚀 How to Use
    1. Go to **📊 Train & Evaluate** — upload your IMDB CSV or use demo data to train the models
    2. Go to **🔍 Predict** — type any movie review and get instant sentiment prediction

    ### 📁 Dataset Format
    Upload a CSV with two columns: `review` (text) and `sentiment` (`positive`/`negative`)
    
    *(IMDB Dataset available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews))*
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Train & Evaluate":
    st.title("📊 Train & Evaluate Models")

    # Data source
    data_source = st.radio(
        "Choose data source:",
        ["📁 Upload CSV", "🧪 Use Demo Data (100 sample reviews)"],
        horizontal=True
    )

    df = None

    if data_source == "📁 Upload CSV":
        uploaded = st.file_uploader(
            "Upload IMDB CSV (columns: `review`, `sentiment`)",
            type=["csv"]
        )
        if uploaded:
            df = pd.read_csv(uploaded)
            sample_size = st.slider("Sample size (rows to use)", 500, min(10000, len(df)), min(3000, len(df)), step=500)
            df = df.sample(sample_size, random_state=42)
            st.success(f"✅ Loaded {len(df)} reviews")
            st.dataframe(df.head(5), use_container_width=True)

    else:
        # Built-in demo data
        demo_reviews = [
            ("This movie was absolutely amazing! The acting was superb and the story kept me on edge.", "positive"),
            ("Terrible film. Complete waste of time. The plot made no sense whatsoever.", "negative"),
            ("One of the best movies I have ever seen. A true masterpiece of cinema.", "positive"),
            ("Awful acting, terrible script, and a boring storyline. Very disappointing.", "negative"),
            ("Loved every single minute of it. Would watch again and again!", "positive"),
            ("Boring and predictable. I fell asleep halfway through the movie.", "negative"),
            ("A beautiful film with great performances from the entire cast.", "positive"),
            ("The worst movie of the year. Total disaster from start to finish.", "negative"),
            ("Brilliant storytelling and incredible visuals. Highly recommend!", "positive"),
            ("Dull, lifeless, and completely forgettable. Not worth watching.", "negative"),
            ("Stunning cinematography and an emotional story that left me speechless.", "positive"),
            ("Painfully bad. The director had no idea what they were doing.", "negative"),
            ("A must-watch for every film lover. Absolutely outstanding!", "positive"),
            ("Confusing, poorly acted, and a complete letdown after the trailer.", "negative"),
            ("Heartwarming and funny. Perfect movie for the whole family.", "positive"),
            ("Dragged on forever with no payoff. Complete waste of two hours.", "negative"),
            ("The performances were incredible and the script was flawless.", "positive"),
            ("Not even worth watching for free. Absolutely terrible.", "negative"),
            ("Genuinely moving and beautifully crafted. Loved it deeply.", "positive"),
            ("Generic, predictable, and utterly unoriginal. Skip this one.", "negative"),
        ] * 5  # 100 reviews

        df = pd.DataFrame(demo_reviews, columns=["review", "sentiment"])
        st.info("Using 100 demo reviews for quick training demo.")
        st.dataframe(df.head(5), use_container_width=True)

    if df is not None:
        if st.button("🚀 Train All 3 Naive Bayes Models", type="primary", use_container_width=True):
            cv, results, best_name = train_models(df)
            st.session_state.cv = cv
            st.session_state.results = results
            st.session_state.best_model = results[best_name]["model"]
            st.session_state.chosen_model_name = best_name
            st.session_state.trained = True
            st.success("✅ Models trained successfully!")

    # Show results if trained
    if st.session_state.trained and st.session_state.results:
        results = st.session_state.results
        st.divider()
        st.subheader("📈 Model Accuracy Comparison")

        # Accuracy bar chart
        names = list(results.keys())
        accs = [results[n]["accuracy"] * 100 for n in names]

        col1, col2, col3 = st.columns(3)
        for col, name, acc in zip([col1, col2, col3], names, accs):
            best_tag = " 🏆" if name == st.session_state.chosen_model_name else ""
            col.metric(f"{name}{best_tag}", f"{acc:.2f}%")

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#4CAF50' if n == st.session_state.chosen_model_name else '#2196F3' for n in names]
        bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Naive Bayes Model Comparison", fontsize=14, fontweight='bold')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{acc:.1f}%", ha='center', va='bottom', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Confusion matrix for best model
        st.divider()
        st.subheader(f"🔢 Confusion Matrix — {st.session_state.chosen_model_name} (Best)")
        best = results[st.session_state.chosen_model_name]
        cm = confusion_matrix(best["y_test"], best["y_pred"])

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"], ax=ax2)
        ax2.set_xlabel("Predicted", fontsize=12)
        ax2.set_ylabel("Actual", fontsize=12)
        ax2.set_title("Confusion Matrix", fontsize=13, fontweight='bold')
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(best["y_test"], best["y_pred"],
                                       target_names=["Negative", "Positive"], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        # Model selector for prediction
        st.divider()
        st.subheader("⚙️ Choose Model for Prediction")
        chosen = st.selectbox("Which model to use in the Predict tab?", list(results.keys()),
                              index=list(results.keys()).index(st.session_state.chosen_model_name))
        if st.button("Set as active model"):
            st.session_state.best_model = results[chosen]["model"]
            st.session_state.chosen_model_name = chosen
            st.success(f"✅ Active model set to **{chosen}**")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.title("🔍 Predict Sentiment")

    if not st.session_state.trained:
        st.warning("⚠️ Please go to **📊 Train & Evaluate** first and train the models.")
    else:
        st.info(f"Active model: **{st.session_state.chosen_model_name}**")
        st.divider()

        review_input = st.text_area(
            "✍️ Enter your movie review below:",
            placeholder="e.g. The cinematography was breathtaking and the story kept me hooked throughout...",
            height=150
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=False)

        if predict_btn and review_input.strip():
            # Preprocess
            cleaned = preprocess(review_input)
            vectorized = st.session_state.cv.transform([cleaned]).toarray()

            model = st.session_state.best_model
            pred = model.predict(vectorized)[0]

            # Show probability if model supports it
            try:
                proba = model.predict_proba(vectorized)[0]
                neg_prob, pos_prob = proba[0] * 100, proba[1] * 100
                has_proba = True
            except Exception:
                has_proba = False

            st.divider()
            if pred == 1:
                st.success("## 😊 Positive Sentiment")
                st.markdown("The model predicts this review expresses a **positive** opinion.")
            else:
                st.error("## 😞 Negative Sentiment")
                st.markdown("The model predicts this review expresses a **negative** opinion.")

            if has_proba:
                col1, col2 = st.columns(2)
                col1.metric("Positive Probability", f"{pos_prob:.1f}%")
                col2.metric("Negative Probability", f"{neg_prob:.1f}%")

                fig, ax = plt.subplots(figsize=(6, 1.5))
                ax.barh(["Sentiment"], [pos_prob], color='#4CAF50', label='Positive')
                ax.barh(["Sentiment"], [neg_prob], left=[pos_prob], color='#F44336', label='Negative')
                ax.set_xlim(0, 100)
                ax.set_xlabel("Probability (%)")
                ax.legend(loc='upper right', fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Show preprocessed text
            with st.expander("🔬 See preprocessed text"):
                st.code(cleaned, language=None)

        elif predict_btn:
            st.warning("Please enter a review first.")

        st.divider()
        st.subheader("🧪 Try these examples")
        examples = [
            "This movie was absolutely brilliant! A masterpiece of modern cinema.",
            "Completely awful. The worst film I have seen in years. Do not waste your time.",
            "Decent movie with some good moments but overall pretty average.",
        ]
        for ex in examples:
            if st.button(f'"{ex[:60]}..."', key=ex):
                st.session_state["prefill"] = ex
                st.rerun()

        if "prefill" in st.session_state:
            review_input = st.session_state.pop("prefill")
