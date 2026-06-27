import streamlit as st

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ─── Load pretrained model (cached so loads only once) ────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Paste any movie review — get instant **Positive / Negative** prediction powered by DistilBERT.")
st.divider()

# Load model with spinner
with st.spinner("⏳ Loading AI model (first time takes ~15 sec)..."):
    classifier = load_model()

st.success("✅ Model ready!", icon="🤖")
st.divider()

# ─── Single Review ─────────────────────────────────────────────────────────────
st.subheader("🔍 Analyze a Review")

review = st.text_area(
    "Enter your movie review:",
    placeholder="e.g. The cinematography was breathtaking and the performances were outstanding...",
    height=160,
    label_visibility="collapsed"
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if review.strip():
        with st.spinner("Analyzing..."):
            result = classifier(review)[0]

        label = result["label"]        # "POSITIVE" or "NEGATIVE"
        score = result["score"] * 100  # confidence %

        st.divider()
        if label == "POSITIVE":
            st.success(f"## 😊 Positive")
        else:
            st.error(f"## 😞 Negative")

        col1, col2 = st.columns(2)
        col1.metric("Prediction", label.capitalize())
        col2.metric("Confidence", f"{score:.1f}%")

        # Confidence bar
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        if label == "POSITIVE":
            ax.barh([""], [score], color="#4CAF50")
            ax.barh([""], [100 - score], left=[score], color="#eee")
        else:
            ax.barh([""], [score], color="#F44336")
            ax.barh([""], [100 - score], left=[score], color="#eee")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence %")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.warning("Please enter a review first.")

# ─── Bulk Analysis ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Analyze Multiple Reviews")
st.caption("Enter one review per line")

bulk_input = st.text_area(
    "Multiple reviews (one per line):",
    placeholder="This movie was amazing!\nTerrible film, waste of time.\nDecent watch, nothing special.",
    height=150,
    label_visibility="collapsed"
)

if st.button("Analyze All", use_container_width=True):
    lines = [l.strip() for l in bulk_input.strip().split("\n") if l.strip()]
    if lines:
        with st.spinner(f"Analyzing {len(lines)} reviews..."):
            results = classifier(lines)

        import pandas as pd
        rows = []
        for text, res in zip(lines, results):
            rows.append({
                "Review": text[:80] + ("..." if len(text) > 80 else ""),
                "Sentiment": "😊 Positive" if res["label"] == "POSITIVE" else "😞 Negative",
                "Confidence": f"{res['score']*100:.1f}%"
            })
        df = pd.DataFrame(rows)

        pos = sum(1 for r in results if r["label"] == "POSITIVE")
        neg = len(results) - pos

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(results))
        col2.metric("Positive 😊", pos)
        col3.metric("Negative 😞", neg)

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Please enter at least one review.")

# ─── Try Examples ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("🧪 Quick Examples")

examples = [
    "Absolutely brilliant! One of the best films I have ever seen. A masterpiece.",
    "Terrible movie. Boring, predictable, and a complete waste of two hours.",
    "Started slow but the ending was incredible. Would recommend to everyone.",
    "The worst acting I have seen in years. Avoid at all costs.",
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(f'"{ex[:45]}..."', key=f"ex_{i}", use_container_width=True):
            with st.spinner("Analyzing..."):
                res = classifier(ex)[0]
            label = res["label"]
            score = res["score"] * 100
            icon = "😊" if label == "POSITIVE" else "😞"
            if label == "POSITIVE":
                st.success(f"{icon} {label.capitalize()} — {score:.1f}%")
            else:
                st.error(f"{icon} {label.capitalize()} — {score:.1f}%")
            st.caption(f'"{ex}"')

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: `distilbert-base-uncased-finetuned-sst-2-english` · Fine-tuned on Stanford Sentiment Treebank (SST-2) · 95%+ accuracy on movie reviews")
