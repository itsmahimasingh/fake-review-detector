import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix
 
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")
 
# ---------- Look and feel: warm brown glow background + cream card + Plus Jakarta Sans font ----------
BACKGROUND = "radial-gradient(ellipse at center, #6B4226 0%, #C49A74 45%, #F3E6D6 100%) fixed"
FONT = "Plus Jakarta Sans"   # change this one name to try another Google font
 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
 
/* warm brown glow background */
.stApp {{ background: {BACKGROUND}; }}
[data-testid="stHeader"] {{ background: transparent; }}
 
/* font everywhere */
html, body, .stMarkdown, .stTextArea textarea, button, p, label, span, div, h1, h2, h3 {{
    font-family: '{FONT}', sans-serif;
}}
h1, h2, h3 {{
    font-weight: 800 !important;
    color: #3B2416 !important;
    letter-spacing: -1px;
}}
h1 {{ font-size: 2.6rem !important; }}
p, label, li {{ color: #5A3A24; }}
 
/* cream cards, a bit transparent: each page is its own card so it can slide */
.block-container {{ padding-top: 3rem !important; }}
 
/* review box */
.stTextArea textarea {{
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid #D2B48C;
    border-radius: 16px;
    color: #3B2416;
}}
 
/* expander and alert boxes */
[data-testid="stExpander"], [data-testid="stAlert"] {{
    background: rgba(255, 255, 255, 0.45);
    border-radius: 16px;
}}
 
/* buttons: rounded pills in brown */
.stButton > button {{
    border-radius: 999px;
    border: 1px solid #D2B48C;
    background: #EAD7C0;
    color: #3B2416;
    font-weight: 600;
    padding: 0.45rem 1.4rem;
}}
.stButton > button:hover {{ background: #D2B48C; border-color: #A87B55; color: #3B2416; }}
.stButton > button[kind="primary"] {{ background: #7A4B2A; border-color: #7A4B2A; color: #FFFFFF; }}
.stButton > button[kind="primary"]:hover {{ background: #5A3420; border-color: #5A3420; color: #FFFFFF; }}
.stButton > button p {{ color: inherit !important; }}   /* button words follow the button's colour */
 
/* progress bar track */
.stProgress > div > div {{ background-color: #EAD7C0; }}
</style>
""", unsafe_allow_html=True)
 
 
# ---------- Load the trained model (saved from Colab) ----------
@st.cache_resource
def load_bundle():
    return joblib.load("detector.joblib")
 
 
bundle = load_bundle()
vec = bundle["vec"]
scaler = bundle["scaler"]
model = bundle["model"]
 
vocab = np.array(vec.get_feature_names_out())
n_vocab = len(vocab)
word_coefs = model.coef_[0][:n_vocab]    # weights for the words part of the model
style_coefs = model.coef_[0][n_vocab:]   # weights for the 8 style features
 
STYLE_NAMES = ["n_words", "avg_word_len", "exclamations", "contractions",
               "first_person", "unique_ratio", "caps_ratio", "rating"]
 
# Plain-English explanation for each style signal, depending on which way it pushed.
# The direction comes from the trained model, not from guesses.
STYLE_TEXT = {
    "n_words":      ("Shorter than a typical genuine review ({n_words:.0f} words)",
                     "Long and detailed ({n_words:.0f} words), like genuine reviews"),
    "avg_word_len": ("Uses shorter, simpler words",
                     "Uses longer, more specific words"),
    "exclamations": ("Calm, template-like tone (few or no exclamation marks)",
                     "Emotional punctuation (!), which is more common in human reviews"),
    "contractions": ("Contraction pattern (don't, it's) matches AI-written reviews",
                     "Contraction pattern (don't, it's) matches human reviews"),
    "first_person": ("Heavy use of I / my / me, which AI-written reviews in the data do more",
                     "Light use of I / my / me, closer to human reviews"),
    "unique_ratio": ("Repeats similar words (less varied vocabulary)",
                     "Varied vocabulary, like genuine reviews"),
    "caps_ratio":   ("Capital-letter pattern matches AI-written reviews",
                     "Capital-letter pattern matches human reviews"),
    "rating":       ("Star rating pattern matches AI-written reviews",
                     "Star rating pattern matches human reviews"),
}
 
 
# ---------- Same style features as in the notebook ----------
def style_features(texts, ratings):
    t = texts.fillna("")
    words = t.str.split()
    return pd.DataFrame({
        "n_words":      words.str.len(),
        "avg_word_len": words.apply(lambda w: np.mean([len(x) for x in w]) if w else 0),
        "exclamations": t.str.count("!"),
        "contractions": t.str.count(r"\w['’]\w"),
        "first_person": t.str.lower().str.count(r"\b(?:i|my|me)\b"),
        "unique_ratio": words.apply(lambda w: len(set(w)) / len(w) if w else 0),
        "caps_ratio":   t.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
        "rating":       ratings.values,
    })
 
 
def analyse(text, rating):
    feats = style_features(pd.Series([text]), pd.Series([rating]))
    scaled = scaler.transform(feats)
    X = hstack([vec.transform([text]), csr_matrix(scaled)]).tocsr()
    prob_fake = float(model.predict_proba(X)[0, 1])
    return prob_fake, feats.iloc[0].to_dict(), scaled[0], X
 
 
def explain(f, scaled, X, top=3, min_effect=0.15):
    fake_signs, real_signs = [], []
 
    # writing-style signals: how much each one pushed THIS review toward fake (+) or real (-)
    contrib = scaled * style_coefs
    order = np.argsort(-np.abs(contrib))
    for i in order:
        if abs(contrib[i]) < min_effect:
            continue
        fake_text, real_text = STYLE_TEXT[STYLE_NAMES[i]]
        if contrib[i] > 0 and len(fake_signs) < top:
            fake_signs.append(fake_text.format(**f))
        elif contrib[i] < 0 and len(real_signs) < top:
            real_signs.append(real_text.format(**f))
 
    # words and phrases the model links to fake or genuine reviews
    row = X[:, :n_vocab]
    if row.nnz:
        contrib = row.data * word_coefs[row.indices]
        order = np.argsort(contrib)[::-1]
        fake_words = [vocab[row.indices[i]] for i in order if contrib[i] > 0][:5]
        real_words = [vocab[row.indices[i]] for i in order[::-1] if contrib[i] < 0][:5]
        if fake_words:
            fake_signs.append("Phrases often seen in AI-written reviews: " + ", ".join(f"“{w}”" for w in fake_words))
        if real_words:
            real_signs.append("Phrases often seen in genuine reviews: " + ", ".join(f"“{w}”" for w in real_words))
 
    return fake_signs, real_signs
 
 
# ---------- Page navigation with the drift (same idea as the skincare app) ----------
# A later page comes up from below and an earlier one comes down from above.
# The page we leave is drawn once more as a "ghost" copy that slides away.
# Only the cards move; the brown background stays still.
 
def page_style(nav):
    forward = nav["forward"] if nav else True
    sign = "" if forward else "-"                      # forward: new page comes up from below
    moves = st.session_state.get("moves", 0)           # new animation name each move, so it replays
    ease = "1.1s cubic-bezier(.45, .05, .2, 1)"        # slow start, long gentle finish
    move_css = ""
    if nav:
        move_css = f"""
[class*="st-key-page_"] {{ animation: arrive {ease}; }}
@keyframes arrive {{ from {{ transform: translateY({sign}100vh) scale(0.97); opacity: 0.4; }}
                     to {{ transform: none; opacity: 1; }} }}
[class*="st-key-leaving_"] {{ position: absolute; top: 3rem; left: 0; right: 0; z-index: 1;
    pointer-events: none; animation: leave {ease} forwards; }}
@keyframes leave {{ to {{ transform: translateY({'-' if forward else ''}100vh) scale(0.97);
                          opacity: 0; visibility: hidden; }} }}
"""
    st.markdown(f"""<style>
[data-testid="stAppViewContainer"] {{ background: transparent; overflow-x: hidden; }}
[data-testid="stMain"], section.main {{ position: relative; z-index: 1; }}
[data-testid="stMainBlockContainer"], .block-container {{ position: relative; }}
[class*="st-key-page_"], [class*="st-key-leaving_"] {{
    background: rgba(250, 243, 234, 0.84); border: 1px solid rgba(226, 204, 178, 0.9);
    border-radius: 28px; padding: 2.5rem 2.5rem 2rem; will-change: transform, opacity; }}
{move_css}
</style>""", unsafe_allow_html=True)
 
 
def go(page):
    forward = page > st.session_state.page
    st.session_state.nav = {"from": st.session_state.page, "forward": forward}
    st.session_state.moves = st.session_state.get("moves", 0) + 1
    st.session_state.page = page
 
 
def run_analysis():
    """Runs when Analyse is clicked: check the review, score it, then drift to the result page."""
    text = st.session_state.get("review", "")
    if len(text.split()) < 3:
        st.session_state.warn = True
        return
    prob, feats, scaled, X = analyse(text, st.session_state.get("rating", 5))
    st.session_state.result = {"text": text, "rating": st.session_state.get("rating", 5),
                               "prob": prob, "feats": feats, "scaled": scaled, "X": X}
    go(2)
 
 
def step(n):
    st.caption(f"Step {n} of 3")
    st.progress(n / 3)
 
 
# ---------- Page 1: Check a review ----------
def page_check(ghost=False):
    g = "ghost_" if ghost else ""
    step(1)
    st.title("🔍 Fake Review Detector")
    st.write("Paste a product review and find out whether it looks genuine or AI-generated.")
    if ghost:   # the copy that slides away shows what was typed
        res = st.session_state.get("result", {})
        st.text_area("Review text", value=res.get("text", ""), key="ghost_review", height=180)
        st.slider("Star rating given in the review", 1, 5, res.get("rating", 5), key="ghost_rating")
    else:
        st.text_area("Review text", key="review", height=180,
                     placeholder="e.g. Best product ever!!! Amazing!!! Must buy!!!")
        st.slider("Star rating given in the review", 1, 5, 5, key="rating")
        if st.session_state.pop("warn", False):
            st.warning("Please paste a review with at least a few words.")
    st.button("Analyse →", key=g + "analyse", type="primary", on_click=run_analysis)
 
 
# ---------- Page 2: Result ----------
def page_result(ghost=False):
    g = "ghost_" if ghost else ""
    res = st.session_state["result"]
    prob = res["prob"]
    step(2)
    st.title("Result")
    if prob >= 0.7:
        st.error(f"### {prob:.0%} likely fake")
        st.write("This review has many signs of being AI-generated.")
    elif prob >= 0.4:
        st.warning(f"### {prob:.0%} likely fake")
        st.write("Mixed signals. Read this review with some caution.")
    else:
        st.success(f"### {prob:.0%} likely fake")
        st.write("This review looks genuine.")
    st.progress(prob)
 
    with st.expander("Review you checked"):
        st.write(res["text"])
 
    c1, c2 = st.columns(2)
    c1.button("← Check another", key=g + "res_back", on_click=go, args=(1,))
    c2.button("See why →", key=g + "res_why", on_click=go, args=(3,), type="primary")
 
 
# ---------- Page 3: Why? ----------
def page_why(ghost=False):
    g = "ghost_" if ghost else ""
    res = st.session_state["result"]
    step(3)
    st.title("Why?")
    fake_signs, real_signs = explain(res["feats"], res["scaled"], res["X"])
 
    st.subheader("🚩 Signs it may be fake")
    if fake_signs:
        for s in fake_signs:
            st.write("• " + s)
    else:
        st.write("No strong warning signs found.")
 
    st.subheader("✅ Signs it looks genuine")
    if real_signs:
        for s in real_signs:
            st.write("• " + s)
    else:
        st.write("No strong genuine signs found.")
 
    st.info("This model was trained on ~40,000 product reviews labelled as human-written "
            "or AI-generated (GPT-2). Treat the result as a helpful guide, not proof.")
 
    c1, c2 = st.columns(2)
    c1.button("← Back to result", key=g + "why_back", on_click=go, args=(2,))
    c2.button("Check another review", key=g + "why_again", on_click=go, args=(1,), type="primary")
 
 
PAGE_FUNCTIONS = {1: page_check, 2: page_result, 3: page_why}
 
if "page" not in st.session_state:
    st.session_state.page = 1
if st.session_state.page > 1 and "result" not in st.session_state:
    st.session_state.page = 1          # no review checked yet, so start at page 1
page = st.session_state.page
 
nav = st.session_state.pop("nav", None)          # set only right after a button click
page_style(nav)
if nav and (nav["from"] == 1 or "result" in st.session_state):
    with st.container(key=f"leaving_{nav['from']}"):     # the old page, sliding away
        PAGE_FUNCTIONS[nav["from"]](ghost=True)
with st.container(key=f"page_{page}"):                   # the new page, sliding in
    PAGE_FUNCTIONS[page]()
 
