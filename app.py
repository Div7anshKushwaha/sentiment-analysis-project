import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="CineScore — Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ─── Custom CSS — Cinematic Dark Theme ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e4d9;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 760px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid #1e1e2a;
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #f0ebe0;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.hero-title span { color: #c9a84c; }
.hero-sub {
    font-size: 15px;
    color: #7a7a8c;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Section labels ── */
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e1e2a;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #111118 !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 10px !important;
    color: #e8e4d9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s !important;
    caret-color: #c9a84c !important;
}
.stTextArea textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.12) !important;
}
.stTextArea textarea::placeholder { color: #3a3a50 !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #c9a84c !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: #e0bc5c !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.25) !important;
}

/* ── Secondary button ── */
.stButton > button:not([kind="primary"]) {
    background: #111118 !important;
    color: #9a96a8 !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #c9a84c !important;
    color: #c9a84c !important;
}

/* ── Result cards ── */
.result-card {
    border-radius: 12px;
    padding: 1.75rem 2rem;
    margin: 1.25rem 0;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card.positive {
    background: linear-gradient(135deg, #0d1f12 0%, #091510 100%);
    border-color: #1a4a28;
}
.result-card.negative {
    background: linear-gradient(135deg, #1f0d0d 0%, #150909 100%);
    border-color: #4a1a1a;
}
.result-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.result-card.positive .result-verdict { color: #4ade80; }
.result-card.negative .result-verdict { color: #f87171; }
.result-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    opacity: 0.6;
    margin-bottom: 1.25rem;
}
.conf-bar-wrap {
    background: #0a0a0f;
    border-radius: 6px;
    height: 6px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}
.positive .conf-bar-fill { background: linear-gradient(90deg, #22c55e, #4ade80); }
.negative .conf-bar-fill { background: linear-gradient(90deg, #ef4444, #f87171); }
.conf-text {
    font-size: 13px;
    color: #7a7a8c;
    margin-top: 0.5rem;
}
.conf-pct {
    font-size: 13px;
    font-weight: 600;
}
.positive .conf-pct { color: #4ade80; }
.negative .conf-pct { color: #f87171; }

/* ── Model badge ── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #111118;
    border: 1px solid #2a2a38;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 11px;
    color: #7a7a8c;
    font-weight: 500;
    letter-spacing: 0.03em;
    margin-bottom: 2rem;
}
.model-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.25rem 0;
}
.stat-box {
    flex: 1;
    background: #111118;
    border: 1px solid #2a2a38;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f0ebe0;
}
.stat-num.positive { color: #4ade80; }
.stat-num.negative { color: #f87171; }
.stat-lbl {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7a7a8c;
    margin-top: 0.2rem;
}

/* ── Example chips ── */
.example-chip {
    background: #111118;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 13px;
    color: #9a96a8;
    cursor: pointer;
    transition: all 0.2s;
    line-height: 1.4;
}
.example-chip:hover {
    border-color: #c9a84c;
    color: #e8e4d9;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #c9a84c !important; }

/* ── Success / error override ── */
.stAlert { border-radius: 10px !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #1e1e2a;
    font-size: 12px;
    color: #3a3a50;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ AI-Powered Analysis</div>
    <div class="hero-title">Cine<span>Score</span></div>
    <div class="hero-sub">Understand the emotion behind every review — instantly</div>
</div>
""", unsafe_allow_html=True)

# Model status badge
with st.spinner("Loading model..."):
    classifier = load_model()

st.markdown("""
<div style="text-align:center">
    <span class="model-badge">
        <span class="model-dot"></span>
        DistilBERT · SST-2 · 95%+ accuracy
    </span>
</div>
""", unsafe_allow_html=True)


# ─── Single Review ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Analyze a review</div>', unsafe_allow_html=True)

review = st.text_area(
    "review_input",
    placeholder="Type or paste a movie review here — the more detail, the better the analysis...",
    height=150,
    label_visibility="collapsed",
    key="main_review"
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if review.strip():
        with st.spinner("Reading the review..."):
            result = classifier(review)[0]

        label = result["label"]
        score = result["score"] * 100
        card_class = "positive" if label == "POSITIVE" else "negative"
        verdict = "Positive" if label == "POSITIVE" else "Negative"
        icon = "✦" if label == "POSITIVE" else "✗"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">{icon} Sentiment verdict</div>
            <div class="result-verdict">{verdict}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span class="conf-text">Confidence</span>
                <span class="conf-pct">{score:.1f}%</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{score}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#c9a84c; font-size:13px; padding:0.5rem 0;">
            ↑ Write something first
        </div>
        """, unsafe_allow_html=True)


# ─── Quick Examples ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:2.5rem">Quick examples</div>', unsafe_allow_html=True)

examples = [
    "A masterpiece. Every frame is deliberate, every performance is lived-in. One of the finest films of the decade.",
    "Unwatchable. Two hours I will never get back. The script makes no sense and the acting is embarrassing.",
    "Slow to start but the final act is genuinely breathtaking. Left the theatre speechless.",
    "Loud, hollow, and forgettable. A corporate product disguised as a movie.",
]

col1, col2 = st.columns(2)
for i, ex in enumerate(examples):
    col = col1 if i % 2 == 0 else col2
    with col:
        if st.button(f'"{ex[:52]}…"', key=f"ex_{i}", use_container_width=True):
            with st.spinner("Analyzing..."):
                res = classifier(ex)[0]
            lbl = res["label"]
            sc = res["score"] * 100
            card_cls = "positive" if lbl == "POSITIVE" else "negative"
            vrd = "Positive" if lbl == "POSITIVE" else "Negative"
            st.markdown(f"""
            <div class="result-card {card_cls}" style="padding:1rem 1.25rem; margin:0.5rem 0;">
                <div class="result-verdict" style="font-size:1.3rem">{vrd}</div>
                <div class="conf-bar-wrap" style="margin-top:0.5rem">
                    <div class="conf-bar-fill" style="width:{sc}%"></div>
                </div>
                <div class="conf-text" style="margin-top:0.35rem">{sc:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Bulk Analysis ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:2.5rem">Bulk analysis</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:13px; color:#7a7a8c; margin-bottom:0.75rem;">One review per line — analyze a whole dataset at once</div>', unsafe_allow_html=True)

bulk_input = st.text_area(
    "bulk",
    placeholder="This movie was stunning...\nTotal disappointment, skip it.\nA slow burn that pays off beautifully.",
    height=130,
    label_visibility="collapsed",
    key="bulk_reviews"
)

if st.button("Analyze All Reviews", use_container_width=True):
    lines = [l.strip() for l in bulk_input.strip().split("\n") if l.strip()]
    if lines:
        with st.spinner(f"Analyzing {len(lines)} reviews..."):
            results = classifier(lines)

        pos = sum(1 for r in results if r["label"] == "POSITIVE")
        neg = len(results) - pos

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-num">{len(results)}</div>
                <div class="stat-lbl">Total</div>
            </div>
            <div class="stat-box">
                <div class="stat-num positive">{pos}</div>
                <div class="stat-lbl">Positive</div>
            </div>
            <div class="stat-box">
                <div class="stat-num negative">{neg}</div>
                <div class="stat-lbl">Negative</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#c9a84c">{pos/len(results)*100:.0f}%</div>
                <div class="stat-lbl">Positive rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        rows = []
        for text, res in zip(lines, results):
            rows.append({
                "Review": text[:90] + ("…" if len(text) > 90 else ""),
                "Sentiment": "✦ Positive" if res["label"] == "POSITIVE" else "✗ Negative",
                "Confidence": f"{res['score']*100:.1f}%"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div style="color:#c9a84c; font-size:13px;">Enter at least one review above.</div>', unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CineScore · distilbert-base-uncased-finetuned-sst-2-english · Stanford SST-2
</div>
""", unsafe_allow_html=True)
