"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Twitter NLP Analysis — Streamlit Dashboard                                ║
║   Run : streamlit run app.py                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Sections                                                                  ║
║   1 · Dataset Overview        EDA plots + statistics                        ║
║   2 · Text Cleaning           Raw vs cleaned tweet comparison               ║
║   3 · WordCloud               Before / After preprocessing                  ║
║   4 · Topic Modeling          LDA topics + keyword explorer                 ║
║   5 · Word2Vec Explorer       Interactive similarity search                 ║
║   6 · Sentiment Prediction    Live tweet → model prediction                 ║
║   7 · Test Results            Predicted distribution + CSV download         ║
╚══════════════════════════════════════════════════════════════════════════════╝
NOTE : Models are loaded from disk — nothing is retrained in this app.
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, re, time, warnings, io
warnings.filterwarnings("ignore")

# ── core ──────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── NLP ───────────────────────────────────────────────────────────────────────
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

for _pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    nltk.download(_pkg, quiet=True)

try:
    import spacy
    try:
        _SPACY = spacy.load("en_core_web_sm")
    except OSError:
        _SPACY = None
except ImportError:
    _SPACY = None

try:
    from wordcloud import WordCloud as _WC
    WC_OK = True
except ImportError:
    WC_OK = False

# ── Gensim ────────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec, LdaModel
    GENSIM_OK = True
except ImportError:
    GENSIM_OK = False

# ── sklearn / joblib ──────────────────────────────────────────────────────────
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
OUT        = "outputs"

IMGS = {
    "eda_dist"    : f"{OUT}/eda_distributions.png",
    "eda_tt"      : f"{OUT}/eda_train_test.png",
    "wc_raw"      : f"{OUT}/wordcloud_raw.png",
    "wc_clean"    : f"{OUT}/wordcloud_clean.png",
    "lda"         : f"{OUT}/lda_topics.png",
    "w2v_heat"    : f"{OUT}/word2vec_similarity.png",
    "confusion"   : f"{OUT}/classification_confusion.png",
    "metrics"     : f"{OUT}/metrics_comparison.png",
    "ner"         : f"{OUT}/ner_analysis.png",
    "pred_dist"   : f"{OUT}/test_predictions_distribution.png",
}

SENTIMENT_MAP = {
    0:"negative", 4:"positive", 1:"positive", -1:"negative",
    "0":"negative","4":"positive","1":"positive","-1":"negative",
    "neg":"negative","pos":"positive",
    "negative":"negative","positive":"positive",
}

CUSTOM_SW = set(stopwords.words("english")) | {
    "rt","amp","http","https","www","com","co","via","im","ur",
    "u","r","got","get","go","like","just","dont","cant","know",
    "think","would","could","really","still","even","much",
    "want","make","day","time","today","said",
}

# Colour palette (consistent across all plots)
PALETTE = {"positive": "#2ecc71", "negative": "#e74c3c"}
PLT_BG   = "#0d1117"
PLT_SURF = "#161b22"
PLT_TEXT = "#e6edf3"
PLT_MUTE = "#8b949e"
PLT_ACC  = "#58a6ff"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ← must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twitter NLP Dashboard",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:#0d1117; --surf:#161b22; --bdr:#21262d;
  --acc:#58a6ff; --acc2:#f78166; --acc3:#3fb950; --acc4:#d2a8ff;
  --txt:#e6edf3; --muted:#8b949e; --rad:12px;
}

html,body,[class*="css"] {
  font-family:'DM Sans',sans-serif;
  background:var(--bg)!important;
  color:var(--txt)!important;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 2.5rem 4rem;max-width:1200px}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--bdr)}
[data-testid="stSidebar"] *{color:var(--txt)!important}
[data-testid="stSidebar"] .stRadio label{
  font-size:.92rem;font-weight:500;padding:.35rem .5rem;
  border-radius:6px;transition:background .15s;
}
[data-testid="stSidebar"] .stRadio label:hover{background:rgba(88,166,255,.12)}

/* Typography */
h1,h2,h3{font-family:'DM Serif Display',serif;letter-spacing:-.02em}
h1{font-size:2.2rem;line-height:1.15}
h2{font-size:1.5rem}
h3{font-size:1.1rem}

/* Cards */
.card{
  background:var(--surf);border:1px solid var(--bdr);
  border-radius:var(--rad);padding:1.3rem 1.5rem;
  margin-bottom:.85rem;transition:border-color .2s,transform .2s;
}
.card:hover{border-color:var(--acc);transform:translateY(-1px)}

/* KPI */
.kpi{background:var(--surf);border:1px solid var(--bdr);border-radius:var(--rad);
     padding:1.2rem 1.4rem;text-align:center;transition:border-color .2s,transform .2s}
.kpi:hover{border-color:var(--acc);transform:translateY(-2px)}
.kpi-n{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:600;
       color:var(--acc);line-height:1}
.kpi-l{font-size:.75rem;color:var(--muted);text-transform:uppercase;
       letter-spacing:.08em;margin-top:.35rem}

/* Labels & pills */
.lbl{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;
     color:var(--muted);font-family:'JetBrains Mono',monospace;margin-bottom:.4rem}
.pill{display:inline-block;background:rgba(210,168,255,.12);border:1px solid var(--acc4);
      color:var(--acc4);border-radius:4px;padding:.18rem .55rem;font-size:.8rem;
      margin:.12rem;font-family:'JetBrains Mono',monospace}
.pill-green{background:rgba(63,185,80,.12);border-color:var(--acc3);color:var(--acc3)}
.pill-red  {background:rgba(247,129,102,.12);border-color:var(--acc2);color:var(--acc2)}
.pill-blue {background:rgba(88,166,255,.12);border-color:var(--acc);color:var(--acc)}

/* Sentiment badges */
.badge-pos{display:inline-block;background:rgba(63,185,80,.15);border:1px solid var(--acc3);
           color:var(--acc3);border-radius:20px;padding:.4rem 1.2rem;
           font-weight:600;font-size:1.1rem}
.badge-neg{display:inline-block;background:rgba(247,129,102,.15);border:1px solid var(--acc2);
           color:var(--acc2);border-radius:20px;padding:.4rem 1.2rem;
           font-weight:600;font-size:1.1rem}

/* NER chips */
.ner{display:inline-block;border-radius:4px;padding:.18rem .5rem;
     font-size:.78rem;margin:.12rem;font-family:'JetBrains Mono',monospace}
.ner-P{background:rgba(88,166,255,.15);border:1px solid var(--acc);color:var(--acc)}
.ner-O{background:rgba(63,185,80,.15);border:1px solid var(--acc3);color:var(--acc3)}
.ner-G{background:rgba(247,129,102,.15);border:1px solid var(--acc2);color:var(--acc2)}
.ner-L{background:rgba(210,168,255,.15);border:1px solid var(--acc4);color:var(--acc4)}

/* W2V rows */
.w2v-row{display:flex;align-items:center;padding:.5rem .8rem;border-bottom:1px solid var(--bdr)}
.w2v-row:last-child{border-bottom:none}
.w2v-word{font-family:'JetBrains Mono',monospace;font-size:.93rem;width:120px}
.w2v-score{font-family:'JetBrains Mono',monospace;color:var(--acc);font-size:.88rem;width:60px;text-align:right}
.w2v-bar-bg{flex:1;margin:0 .8rem;background:var(--bdr);border-radius:3px;height:5px}
.w2v-bar-fg{height:5px;border-radius:3px;background:var(--acc)}

/* Diff table */
.diff-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:.5rem}
.diff-cell{background:var(--surf);border:1px solid var(--bdr);border-radius:8px;
           padding:.8rem 1rem;font-size:.88rem;line-height:1.5}
.diff-hdr{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
          color:var(--muted);font-family:'JetBrains Mono',monospace;margin-bottom:.4rem}

/* Metric table */
.mtbl{width:100%;border-collapse:collapse}
.mtbl th{background:rgba(88,166,255,.1);color:var(--acc);font-family:'JetBrains Mono',monospace;
         font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;
         padding:.55rem .9rem;text-align:left;border-bottom:1px solid var(--bdr)}
.mtbl td{padding:.55rem .9rem;border-bottom:1px solid var(--bdr);
         font-size:.88rem;font-family:'JetBrains Mono',monospace}
.mtbl tr:last-child td{border-bottom:none}
.g{color:var(--acc3);font-weight:600}

/* Divider */
.div{border:none;border-top:1px solid var(--bdr);margin:1.8rem 0}

/* Page header */
.ph{border-left:4px solid var(--acc);padding-left:1rem;margin-bottom:2rem}
.ph p{color:var(--muted);font-size:.93rem;margin-top:.2rem}

/* Section title */
.st-sec{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;
        color:var(--muted);font-family:'JetBrains Mono',monospace;margin:1.4rem 0 .6rem}

/* Inputs */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{
  background:var(--surf)!important;border:1px solid var(--bdr)!important;
  border-radius:8px!important;color:var(--txt)!important}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{
  border-color:var(--acc)!important;box-shadow:0 0 0 3px rgba(88,166,255,.15)!important}
.stButton>button{
  background:var(--acc)!important;color:#0d1117!important;border:none!important;
  border-radius:8px!important;font-weight:600!important;padding:.45rem 1.5rem!important;
  transition:opacity .15s!important}
.stButton>button:hover{opacity:.85!important}

/* selectbox */
.stSelectbox>div>div{background:var(--surf)!important;border:1px solid var(--bdr)!important;
                     border-radius:8px!important;color:var(--txt)!important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME helper
# ─────────────────────────────────────────────────────────────────────────────
def _mpl_dark():
    plt.rcParams.update({
        "figure.facecolor" : PLT_BG,
        "axes.facecolor"   : PLT_SURF,
        "axes.edgecolor"   : PLT_MUTE,
        "axes.labelcolor"  : PLT_TEXT,
        "xtick.color"      : PLT_MUTE,
        "ytick.color"      : PLT_MUTE,
        "text.color"       : PLT_TEXT,
        "grid.color"       : "#21262d",
        "grid.alpha"       : 0.5,
        "figure.dpi"       : 130,
        "font.family"      : "DejaVu Sans",
    })


def _buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=PLT_BG, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_train() -> pd.DataFrame:
    if not os.path.exists(TRAIN_PATH):
        return pd.DataFrame()
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8", on_bad_lines="skip")
    for c in df.columns:
        if c.lower() in ("tweet","text","tweets","texts") and c != "tweet":
            df.rename(columns={c:"tweet"}, inplace=True); break
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].map(
            lambda x: SENTIMENT_MAP.get(x, SENTIMENT_MAP.get(str(x), str(x))))
        df = df[df["sentiment"].isin(["negative","positive"])]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_test() -> pd.DataFrame:
    if not os.path.exists(TEST_PATH):
        return pd.DataFrame()
    df = pd.read_csv(TEST_PATH, encoding="utf-8", on_bad_lines="skip")
    for c in df.columns:
        if c.lower() in ("tweet","text","tweets","texts") and c != "tweet":
            df.rename(columns={c:"tweet"}, inplace=True); break
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = f"{OUT}/test_predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_w2v():
    p = f"{OUT}/word2vec_model"
    if GENSIM_OK and os.path.exists(p):
        try: return Word2Vec.load(p)
        except: pass
    return None


@st.cache_resource(show_spinner=False)
def load_lda():
    p = f"{OUT}/lda_model"
    if GENSIM_OK and os.path.exists(p):
        try: return LdaModel.load(p)
        except: pass
    return None


@st.cache_resource(show_spinner=False)
def load_svm():
    for fn in ["svm_pipeline.pkl","svm.pkl","classifier.pkl","model.pkl"]:
        p = f"{OUT}/{fn}"
        if os.path.exists(p):
            try: return joblib.load(p), fn
            except: pass
    return None, None


@st.cache_resource(show_spinner=False)
def load_mlp():
    for fn in ["mlp_pipeline.pkl","mlp.pkl"]:
        p = f"{OUT}/{fn}"
        if os.path.exists(p):
            try: return joblib.load(p), fn
            except: pass
    return None, None


@st.cache_resource(show_spinner=False)
def load_tfidf():
    for fn in ["tfidf.pkl","tfidf_vectorizer.pkl","vectorizer.pkl"]:
        p = f"{OUT}/{fn}"
        if os.path.exists(p):
            try: return joblib.load(p)
            except: pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NLP HELPERS  (no retraining — just preprocessing + inference)
# ─────────────────────────────────────────────────────────────────────────────
_TK = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


def clean_tweet(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^a-zA-Z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = _TK.tokenize(t)
    tokens = [x for x in tokens if x not in CUSTOM_SW and len(x) > 2]
    return " ".join(tokens)


def lemmatize(text: str) -> str:
    if _SPACY is None:
        return text
    allowed = {"NOUN","VERB","ADJ","ADV"}
    doc = _SPACY(text, disable=["parser","ner"])
    return " ".join(
        tok.lemma_ for tok in doc
        if tok.pos_ in allowed and not tok.is_stop and len(tok.lemma_) > 2
    )


def get_ner(text: str) -> list:
    if _SPACY is None: return []
    doc = _SPACY(text)
    return [(e.text, e.label_) for e in doc.ents
            if e.label_ in {"PERSON","ORG","GPE","LOC","NORP"}]


def lda_predict(text: str, lda):
    if lda is None: return None, []
    bow = lda.id2word.doc2bow(text.split())
    if not bow: return None, []
    topics = sorted(lda.get_document_topics(bow, minimum_probability=0),
                    key=lambda x: x[1], reverse=True)
    tid = topics[0][0]
    return tid, [w for w,_ in lda.show_topic(tid, topn=6)]


def predict_sentiment(text: str, model, tfidf) -> str:
    """Run sklearn pipeline or (model+vectorizer) prediction."""
    if model is None:
        return _lexicon_fallback(text)
    try:
        # If it's a Pipeline object, it already has tfidf internally
        if hasattr(model, "predict"):
            try:
                return model.predict([text])[0]
            except Exception:
                pass
        # Separate tfidf + model
        if tfidf is not None:
            vec = tfidf.transform([text])
            return model.predict(vec)[0]
    except Exception:
        pass
    return _lexicon_fallback(text)


def _lexicon_fallback(text: str) -> str:
    pos = {"love","great","amazing","good","happy","excellent","awesome",
           "wonderful","fantastic","nice","best","super","joy","win","perfect"}
    neg = {"hate","bad","terrible","awful","worst","sad","angry","horrible",
           "disappointing","fail","boring","ugly","poor","sucks","worst"}
    words = set(text.lower().split())
    return "positive" if len(words & pos) >= len(words & neg) else "negative"


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS  (generate matplotlib figures, return buffer)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentiment_dist(df: pd.DataFrame) -> io.BytesIO:
    _mpl_dark()
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [PALETTE.get(s, PLT_ACC) for s in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor="#0d1117", linewidth=0.8, width=0.5)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+counts.max()*.01,
                f"{v:,}", ha="center", va="bottom", fontsize=9, color=PLT_TEXT)
    ax.set_title("Sentiment Distribution", fontsize=11, color=PLT_TEXT, pad=8)
    ax.set_ylabel("Count", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _buf(fig)


def plot_length_hist(df: pd.DataFrame) -> io.BytesIO:
    _mpl_dark()
    df["_len"] = df["tweet"].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(df["_len"], bins=60, color=PLT_ACC, edgecolor="#0d1117", alpha=0.85)
    ax.set_title("Tweet Length (chars)", fontsize=11, color=PLT_TEXT, pad=8)
    ax.set_xlabel("Characters", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _buf(fig)


def plot_wordcount_box(df: pd.DataFrame) -> io.BytesIO:
    _mpl_dark()
    df["_wc"] = df["tweet"].astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    groups = [df[df["sentiment"]==s]["_wc"].dropna()
              for s in ["negative","positive"] if s in df["sentiment"].values]
    labels = [s for s in ["negative","positive"] if s in df["sentiment"].values]
    colors = [PALETTE[s] for s in labels]
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="#ffffff", linewidth=1.5),
                    whiskerprops=dict(color=PLT_MUTE),
                    capprops=dict(color=PLT_MUTE),
                    flierprops=dict(marker="o", color=PLT_MUTE, markersize=2, alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_title("Word Count by Sentiment", fontsize=11, color=PLT_TEXT, pad=8)
    ax.set_ylabel("Words", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _buf(fig)


def plot_top_words(df: pd.DataFrame, col: str = "tweet", n: int = 15) -> io.BytesIO:
    from collections import Counter
    _mpl_dark()
    sample = df[col].dropna().sample(min(30_000, len(df)), random_state=42)
    words  = []
    for t in sample:
        words.extend(clean_tweet(t).split())
    top = Counter(words).most_common(n)
    wds, cnts = zip(*top) if top else ([], [])

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(list(wds)[::-1], list(cnts)[::-1],
                   color=PLT_ACC, edgecolor="#0d1117", alpha=0.85)
    ax.set_title(f"Top {n} Words (cleaned)", fontsize=11, color=PLT_TEXT, pad=8)
    ax.set_xlabel("Frequency", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _buf(fig)


def plot_wc_live(text_series, cmap="plasma", bg="black") -> io.BytesIO:
    if not WC_OK:
        return None
    _mpl_dark()
    text = " ".join(text_series.dropna().sample(min(20_000, len(text_series)), random_state=42))
    wc = _WC(width=900, height=450, background_color=bg,
              colormap=cmap, max_words=200, collocations=False,
              stopwords=CUSTOM_SW).generate(text)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor(PLT_BG)
    fig.tight_layout(pad=0)
    return _buf(fig)


def plot_pred_dist(pred_df: pd.DataFrame) -> io.BytesIO:
    _mpl_dark()
    counts = pred_df["predicted_sentiment"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Bar
    colors = [PALETTE.get(s, PLT_ACC) for s in counts.index]
    bars = ax1.bar(counts.index, counts.values, color=colors,
                   edgecolor="#0d1117", width=0.45)
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+counts.max()*.01,
                 f"{v:,}", ha="center", va="bottom", fontsize=9, color=PLT_TEXT)
    ax1.set_title("Prediction Distribution (test.csv)", fontsize=10, color=PLT_TEXT)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Pie
    wedge_colors = [PALETTE.get(s, PLT_ACC) for s in counts.index]
    ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=wedge_colors, startangle=90,
            wedgeprops=dict(edgecolor="#0d1117", linewidth=1),
            textprops=dict(color=PLT_TEXT, fontsize=9))
    ax2.set_title("Sentiment Split", fontsize=10, color=PLT_TEXT)

    fig.patch.set_facecolor(PLT_BG)
    fig.tight_layout()
    return _buf(fig)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def show_img(key: str, caption: str = "", width=None):
    path = IMGS.get(key, key)
    if os.path.exists(path):
        kw = {"use_container_width": True} if width is None else {"width": width}
        st.image(path, caption=caption or None, **kw)
    else:
        st.info(f"Image not found: `{path}`  — run `code.py` first.")


def kpi(col, number, label):
    with col:
        st.markdown(f"""
        <div class='kpi'>
          <div class='kpi-n'>{number}</div>
          <div class='kpi-l'>{label}</div>
        </div>""", unsafe_allow_html=True)


def section(title: str):
    st.markdown(f"<div class='st-sec'>{title}</div>", unsafe_allow_html=True)


def divider():
    st.markdown("<hr class='div'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PRE-LOAD DATA  (so sidebar stats are always available)
# ─────────────────────────────────────────────────────────────────────────────
train_df   = load_train()
test_df    = load_test()
pred_df    = load_predictions()
lda_model  = load_lda()
w2v_model  = load_w2v()
svm_model, svm_name = load_svm()
mlp_model, mlp_name = load_mlp()
tfidf_vec  = load_tfidf()

total = len(train_df) + len(test_df)
n_pos = len(train_df[train_df["sentiment"]=="positive"]) if "sentiment" in train_df.columns else 0
n_neg = len(train_df[train_df["sentiment"]=="negative"]) if "sentiment" in train_df.columns else 0

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 1.2rem'>
      <div style='font-family:"DM Serif Display",serif;font-size:1.35rem;
                  color:#e6edf3;line-height:1.2'>🐦 Twitter NLP</div>
      <div style='font-size:.72rem;color:#8b949e;margin-top:.2rem;
                  font-family:"JetBrains Mono",monospace'>Analysis Dashboard</div>
    </div>
    <hr style='border-color:#21262d;margin-bottom:1rem'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊  Dataset Overview",
        "🧹  Text Cleaning",
        "☁️  WordCloud",
        "🗂️  Topic Modeling",
        "🔤  Word2Vec Explorer",
        "🔍  Sentiment Prediction",
        "📁  Test Results",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#21262d;margin:1rem 0'>", unsafe_allow_html=True)

    # Quick stats
    st.markdown(f"""
    <div style='font-size:.73rem;color:#8b949e;font-family:"JetBrains Mono",monospace;line-height:2'>
      <div>🗂  train  <span style='color:#e6edf3'>{len(train_df):,}</span></div>
      <div>🗂  test   <span style='color:#e6edf3'>{len(test_df):,}</span></div>
      <div>😊  pos    <span style='color:#3fb950'>{n_pos:,}</span></div>
      <div>😡  neg    <span style='color:#f78166'>{n_neg:,}</span></div>
      <div>🗺️  topics <span style='color:#d2a8ff'>{lda_model.num_topics if lda_model else "N/A"}</span></div>
      <div>📐  vocab  <span style='color:#58a6ff'>{len(w2v_model.wv) if w2v_model else "N/A"}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:2rem;font-size:.68rem;color:#8b949e;text-align:center'>
      SG01 · University NLP Project<br>1.6M Tweets · NLTK · spaCy · Gensim
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊  Dataset Overview":
    st.markdown("""
    <div class='ph'><h1>Dataset Overview</h1>
    <p>Exploratory data analysis · 1.6M tweets · Sentiment140 (Kaggle)</p></div>
    """, unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, f"{total:,}",    "Total Tweets")
    kpi(c2, f"{len(train_df):,}", "Train")
    kpi(c3, f"{len(test_df):,}",  "Test (unlabeled)")
    kpi(c4, f"{n_pos:,}",    "😊 Positive")
    kpi(c5, f"{n_neg:,}",    "😡 Negative")

    divider()
    section("Distribution Plots")

    if not train_df.empty and "sentiment" in train_df.columns:
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            with st.spinner("Sentiment bar…"):
                st.image(plot_sentiment_dist(train_df), use_container_width=True)
        with pc2:
            with st.spinner("Length histogram…"):
                st.image(plot_length_hist(train_df), use_container_width=True)
        with pc3:
            with st.spinner("Boxplot…"):
                st.image(plot_wordcount_box(train_df), use_container_width=True)
    else:
        show_img("eda_dist", "Sentiment / Length distributions")

    divider()
    section("Train vs Test Split")
    cc1, cc2 = st.columns(2)
    with cc1:
        show_img("eda_tt", "Train vs Test breakdown")
    with cc2:
        section("Top 15 Words (cleaned corpus)")
        if not train_df.empty:
            with st.spinner("Computing top words…"):
                st.image(plot_top_words(train_df), use_container_width=True)

    divider()
    section("NER Analysis")
    show_img("ner", "Named Entity Recognition (spaCy — train sample)")

    divider()
    section("Sample Data")
    tab1, tab2 = st.tabs(["Train sample", "Test sample"])
    with tab1:
        if not train_df.empty:
            cols = [c for c in ["tweet","sentiment"] if c in train_df.columns]
            st.dataframe(train_df[cols].sample(min(10,len(train_df)),
                         random_state=1).reset_index(drop=True),
                         use_container_width=True, height=280)
    with tab2:
        if not test_df.empty:
            cols = [c for c in ["tweet"] if c in test_df.columns]
            st.dataframe(test_df[cols].sample(min(10,len(test_df)),
                         random_state=1).reset_index(drop=True),
                         use_container_width=True, height=280)

    # Descriptive stats
    divider()
    section("Descriptive Statistics")
    if not train_df.empty:
        tmp = train_df.copy()
        tmp["chars"] = tmp["tweet"].astype(str).apply(len)
        tmp["words"] = tmp["tweet"].astype(str).apply(lambda x: len(x.split()))
        st.dataframe(tmp[["chars","words"]].describe().round(2),
                     use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TEXT CLEANING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧹  Text Cleaning":
    st.markdown("""
    <div class='ph'><h1>Text Cleaning Visualization</h1>
    <p>Raw tweet → NLTK cleaning → spaCy lemmatization · Side-by-side comparison</p></div>
    """, unsafe_allow_html=True)

    # Live demo
    section("Live Pipeline Demo")
    demo_input = st.text_area(
        "Enter a tweet to process through the pipeline",
        value="I absolutely LOVE this new AI product!! It's amazing 😍 #MachineLearning @OpenAI https://example.com",
        height=80,
    )

    if st.button("▶  Run Pipeline", key="clean_btn"):
        with st.spinner("Processing…"):
            cleaned = clean_tweet(demo_input)
            lemma   = lemmatize(cleaned)

        st.markdown(f"""
        <div class='diff-row'>
          <div class='diff-cell'>
            <div class='diff-hdr'>01 · Original</div>
            {demo_input}
          </div>
          <div class='diff-cell'>
            <div class='diff-hdr'>02 · After NLTK cleaning</div>
            <span style='font-family:"JetBrains Mono",monospace;
                         font-size:.85rem;color:#8b949e'>{cleaned or "<em>empty</em>"}</span>
          </div>
        </div>
        <div class='diff-row'>
          <div class='diff-cell'>
            <div class='diff-hdr'>03 · After spaCy lemmatization</div>
            <span style='font-family:"JetBrains Mono",monospace;
                         font-size:.85rem;color:#d2a8ff'>{lemma or "<em>empty</em>"}</span>
          </div>
          <div class='diff-cell'>
            <div class='diff-hdr'>04 · Tokens removed</div>
            <span style='font-size:.85rem;color:#8b949e'>
              {max(0, len(demo_input.split()) - len(cleaned.split()))} tokens removed
              · {len(cleaned.split())} tokens remain
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.success("Pipeline complete!", icon="✅")

    divider()
    section("Random Samples from Train (raw vs cleaned)")

    if not train_df.empty:
        n_show = st.slider("Number of samples", 3, 15, 6)
        sample = train_df["tweet"].dropna().sample(n_show, random_state=42)
        rows_html = ""
        for raw in sample:
            cl = clean_tweet(raw)
            rows_html += f"""
            <div class='diff-row' style='margin-bottom:.3rem'>
              <div class='diff-cell'>
                <div class='diff-hdr'>Raw</div>{raw[:200]}
              </div>
              <div class='diff-cell'>
                <div class='diff-hdr'>Cleaned</div>
                <span style='font-family:"JetBrains Mono",monospace;
                             font-size:.83rem;color:#8b949e'>{cl or "<em>—</em>"}</span>
              </div>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)

    divider()
    section("Pipeline Steps Explained")
    steps = [
        ("1 · Lowercasing", "Convert all text to lowercase to normalise case."),
        ("2 · URL removal", "Strip http/https links — they carry no sentiment signal."),
        ("3 · Mention removal", "Remove @username handles (@OpenAI → removed)."),
        ("4 · Hashtag deconstruction", "#MachineLearning → MachineLearning."),
        ("5 · Punctuation strip", "Remove special characters, keep apostrophes."),
        ("6 · Tokenization (NLTK TweetTokenizer)", "Split into tokens, reduce lengthening (loooove → love)."),
        ("7 · Stopword removal", f"Remove {len(CUSTOM_SW)} custom stopwords (NLTK + Twitter-specific)."),
        ("8 · Lemmatization (spaCy)", "love/loved/loves → love. Filter by POS: NOUN, VERB, ADJ, ADV."),
    ]
    for title, desc in steps:
        st.markdown(f"""
        <div class='card' style='padding:.9rem 1.2rem'>
          <span style='font-family:"JetBrains Mono",monospace;font-size:.82rem;
                       color:#58a6ff'>{title}</span>
          <span style='color:#8b949e;font-size:.88rem;margin-left:.8rem'>{desc}</span>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WORDCLOUD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "☁️  WordCloud":
    st.markdown("""
    <div class='ph'><h1>WordCloud Comparison</h1>
    <p>Before and after text preprocessing — spot the noise reduction</p></div>
    """, unsafe_allow_html=True)

    # Toggle: precomputed vs live
    mode = st.radio("Display mode", ["Precomputed (fast)", "Generate live (slower)"],
                    horizontal=True, label_visibility="collapsed")
    st.caption("▲ Choose display mode")

    divider()

    if mode == "Precomputed (fast)":
        c1, c2 = st.columns(2)
        with c1:
            section("Before Cleaning")
            show_img("wc_raw")
        with c2:
            section("After Cleaning & Lemmatization")
            show_img("wc_clean")
    else:
        if not train_df.empty and WC_OK:
            cmap_choice = st.selectbox("Colormap", ["plasma","viridis","inferno","magma","cool"])
            col1, col2 = st.columns(2)
            with col1:
                section("Before Cleaning (raw tweets)")
                with st.spinner("Generating raw WordCloud…"):
                    buf = plot_wc_live(train_df["tweet"], cmap=cmap_choice, bg="black")
                    if buf: st.image(buf, use_container_width=True)
            with col2:
                section("After Cleaning")
                with st.spinner("Cleaning + generating WordCloud…"):
                    cleaned_series = train_df["tweet"].dropna().apply(clean_tweet)
                    buf2 = plot_wc_live(cleaned_series, cmap="viridis", bg="white")
                    if buf2: st.image(buf2, use_container_width=True)
        elif not WC_OK:
            st.warning("wordcloud package not installed. Showing precomputed images.")
            show_img("wc_raw", "Raw")
            show_img("wc_clean", "Clean")
        else:
            st.info("train.csv not found.")

    divider()
    section("Key Observations")
    obs = [
        ("🔴 Raw corpus",  "URLs, mentions (@user), 'http', 'rt', 'amp' dominate."),
        ("🟢 After cleaning", "Meaningful words: love, good, day, feel, work, happy…"),
        ("🔵 Lemmatization effect", "Inflected forms merged: 'loved' / 'loves' → 'love'."),
        ("📉 Vocabulary reduction", "~40% vocabulary reduction after stopword removal."),
    ]
    for icon_title, text in obs:
        st.markdown(f"""
        <div class='card' style='padding:.85rem 1.2rem'>
          <span style='font-weight:600'>{icon_title}</span>
          <span style='color:#8b949e;margin-left:.6rem;font-size:.9rem'>{text}</span>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TOPIC MODELING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗂️  Topic Modeling":
    st.markdown("""
    <div class='ph'><h1>Topic Modeling — LDA</h1>
    <p>Latent Dirichlet Allocation · Gensim · Trained on 50K tweets</p></div>
    """, unsafe_allow_html=True)

    # LDA visualisation image
    section("Topic Weight Distribution")
    show_img("lda", "Top keywords per topic (weight)")

    divider()
    section("Topic Keywords Explorer")

    if lda_model is not None:
        n_topics = lda_model.num_topics

        # Interactive per-topic view
        col_sel, col_detail = st.columns([1, 3])
        with col_sel:
            chosen = st.selectbox(
                "Select a topic",
                [f"Topic {i+1}" for i in range(n_topics)],
            )
            n_words = st.slider("Keywords to show", 5, 20, 10)

        tid = int(chosen.split()[1]) - 1
        pairs = lda_model.show_topic(tid, topn=n_words)

        with col_detail:
            # Bar chart
            _mpl_dark()
            fig, ax = plt.subplots(figsize=(7, 3.2))
            words  = [w for w,_ in pairs]
            scores = [s for _,s in pairs]
            palette_t = plt.cm.plasma(np.linspace(0.3, 0.9, len(words)))
            bars = ax.barh(words[::-1], scores[::-1], color=palette_t[::-1],
                           edgecolor="#0d1117", linewidth=0.6)
            ax.set_title(f"Topic {tid+1} — keyword weights", fontsize=10,
                         color=PLT_TEXT, pad=6)
            ax.set_xlabel("Weight", fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            ax.xaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.image(_buf(fig), use_container_width=True)

        # Pills
        pills = "".join(f"<span class='pill'>{w}</span>" for w,_ in pairs)
        st.markdown(f"<div style='margin-top:.5rem'>{pills}</div>", unsafe_allow_html=True)

        divider()
        section("All Topics at a Glance")
        for i in range(n_topics):
            kws  = lda_model.show_topic(i, topn=8)
            lead = kws[0][0] if kws else "—"
            p    = "".join(f"<span class='pill'>{w}</span>" for w,_ in kws)
            st.markdown(f"""
            <div class='card' style='padding:.9rem 1.2rem'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem'>
                <span style='font-family:"JetBrains Mono",monospace;color:#58a6ff;font-weight:600'>
                  Topic {i+1}</span>
                <span style='font-size:.72rem;color:#8b949e'>lead: {lead}</span>
              </div>
              {p}
            </div>""", unsafe_allow_html=True)
    else:
        st.info("LDA model not found. Run `code.py` to generate `outputs/lda_model`.")
        # Fallback static image
        show_img("lda")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — WORD2VEC EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔤  Word2Vec Explorer":
    st.markdown("""
    <div class='ph'><h1>Word2Vec Explorer</h1>
    <p>Skip-gram embeddings · 100D · Trained on 50K tweets · Semantic neighborhood search</p></div>
    """, unsafe_allow_html=True)

    left, right = st.columns([2, 3])

    with left:
        section("Search")
        word_in = st.text_input("Enter a word", placeholder="e.g. love, twitter, happy …")
        top_n   = st.slider("Similar words to return", 3, 20, 8)
        algo    = st.radio("Similarity metric", ["cosine (default)", "euclidean (approx)"],
                           horizontal=True)
        go_btn  = st.button("🔍  Find Similar Words", use_container_width=True)

    with right:
        section("Cosine Similarity Heatmap (probe words)")
        show_img("w2v_heat")

    divider()

    if go_btn or word_in:
        word_q = word_in.strip().lower()
        if not word_q:
            st.warning("Please enter a word.", icon="⚠️")
        elif w2v_model is None:
            st.error("Word2Vec model not found. Place `outputs/word2vec_model` in outputs/.", icon="❌")
        elif word_q not in w2v_model.wv:
            st.error(f"**'{word_q}'** not in vocabulary ({len(w2v_model.wv):,} words). "
                     "Try a simpler/more common word.", icon="❌")
        else:
            with st.spinner(f"Finding neighbors of '{word_q}' …"):
                similar = w2v_model.wv.most_similar(word_q, topn=top_n)

            st.markdown(f"""
            <div style='margin-bottom:.8rem'>
              <span style='font-family:"DM Serif Display",serif;font-size:1.1rem'>
                Nearest neighbors of</span>
              <span style='font-family:"JetBrains Mono",monospace;font-size:1rem;
                           color:#58a6ff;margin-left:.4rem'>"{word_q}"</span>
            </div>""", unsafe_allow_html=True)

            max_s = similar[0][1] if similar else 1.0
            rows  = ""
            for rank, (w, s) in enumerate(similar, 1):
                pct = int(s / max_s * 100)
                rows += f"""
                <div class='w2v-row'>
                  <span style='color:#8b949e;font-family:"JetBrains Mono",monospace;
                               font-size:.78rem;width:20px'>#{rank}</span>
                  <span class='w2v-word'>&nbsp;{w}</span>
                  <div class='w2v-bar-bg'><div class='w2v-bar-fg' style='width:{pct}%'></div></div>
                  <span class='w2v-score'>{s:.4f}</span>
                </div>"""

            st.markdown(f"<div class='card' style='padding:.4rem'>{rows}</div>",
                        unsafe_allow_html=True)

            # Mini bar chart
            _mpl_dark()
            fig, ax = plt.subplots(figsize=(6, 3))
            wds  = [w for w,_ in similar]
            scos = [s for _,s in similar]
            pal  = plt.cm.Blues(np.linspace(0.4, 0.85, len(wds)))
            ax.barh(wds[::-1], scos[::-1], color=pal, edgecolor="#0d1117", linewidth=0.5)
            ax.set_title(f"Similarity scores — '{word_q}'", fontsize=10, color=PLT_TEXT, pad=6)
            ax.set_xlabel("Cosine similarity", fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            ax.xaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.image(_buf(fig), use_container_width=True)

            st.info(f"Vocabulary: **{len(w2v_model.wv):,}** words · "
                    f"Dimensions: **{w2v_model.vector_size}D** · Algorithm: **Skip-gram**")

    divider()
    section("Quick Presets")
    presets = ["love","hate","twitter","music","happy","angry","obama","apple","work","news"]
    cols    = st.columns(len(presets))
    for i, pw in enumerate(presets):
        with cols[i]:
            if st.button(pw, key=f"p_{pw}", use_container_width=True):
                st.session_state["_w2v_word"] = pw
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SENTIMENT PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Sentiment Prediction":
    st.markdown("""
    <div class='ph'><h1>Sentiment Prediction</h1>
    <p>Live tweet → NLP pipeline → sentiment · topic · named entities</p></div>
    """, unsafe_allow_html=True)

    # Model selector
    model_choice = st.radio(
        "Classifier",
        ["SVM (LinearSVC)", "MLP (Neural Net)", "Best available"],
        horizontal=True,
    )

    tweet_in = st.text_area(
        "Enter your tweet",
        placeholder="Type a tweet and click Analyze …",
        height=90,
    )

    analyze  = st.button("🚀  Analyze", use_container_width=False)

    if analyze and tweet_in.strip():
        with st.spinner("Running NLP pipeline …"):
            t0       = time.time()
            cleaned  = clean_tweet(tweet_in)
            lemma    = lemmatize(cleaned)
            entities = get_ner(tweet_in)
            tid, kws = lda_predict(lemma, lda_model)

            # Choose model
            if model_choice == "SVM (LinearSVC)":
                active_model = svm_model
            elif model_choice == "MLP (Neural Net)":
                active_model = mlp_model
            else:
                active_model = svm_model or mlp_model

            sentiment = predict_sentiment(cleaned, active_model, tfidf_vec)
            elapsed   = time.time() - t0

        # ── Results layout ──────────────────────────────────────────────────
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown(f"""
            <div class='card'>
              <div class='lbl'>01 · Original tweet</div>
              <div style='font-size:.95rem;line-height:1.6'>{tweet_in}</div>
            </div>
            <div class='card'>
              <div class='lbl'>02 · After NLTK cleaning</div>
              <div style='font-family:"JetBrains Mono",monospace;font-size:.85rem;
                          color:#8b949e;line-height:1.55'>{cleaned or "<em>empty after cleaning</em>"}</div>
            </div>
            <div class='card'>
              <div class='lbl'>03 · After spaCy lemmatization</div>
              <div style='font-family:"JetBrains Mono",monospace;font-size:.85rem;
                          color:#d2a8ff;line-height:1.55'>{lemma or "<em>empty after lemmatization</em>"}</div>
            </div>
            """, unsafe_allow_html=True)

        with right_col:
            badge_cls = "badge-pos" if sentiment == "positive" else "badge-neg"
            emoji     = "😊" if sentiment == "positive" else "😡"
            model_lbl = svm_name or mlp_name or "heuristic"

            st.markdown(f"""
            <div class='card' style='text-align:center;padding:1.8rem 1rem'>
              <div class='lbl'>04 · Sentiment</div>
              <div style='margin:.7rem 0'>
                <span class='{badge_cls}'>{emoji} {sentiment.capitalize()}</span>
              </div>
              <div style='font-size:.72rem;color:#8b949e;margin-top:.5rem'>
                via {model_lbl} · {elapsed*1000:.0f}ms
              </div>
            </div>
            """, unsafe_allow_html=True)

            # LDA topic
            if tid is not None:
                pills = "".join(f"<span class='pill'>{w}</span>" for w in kws)
                st.markdown(f"""
                <div class='card'>
                  <div class='lbl'>05 · LDA Topic #{tid+1}</div>
                  <div style='margin-top:.4rem'>{pills}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='card'>
                  <div class='lbl'>05 · LDA Topic</div>
                  <div style='color:#8b949e;font-size:.88rem'>
                    LDA model not loaded.</div>
                </div>""", unsafe_allow_html=True)

            # NER
            if entities:
                ner_map = {"PERSON":"P","ORG":"O","GPE":"G","LOC":"L","NORP":"G"}
                chips_parts = []
                for txt, lbl in entities:
                    cls = ner_map.get(lbl, "L")
                    chips_parts.append(
                        f"<span class='ner ner-{cls}'>{txt} "
                        f"<span style='opacity:.6;font-size:.68rem'>[{lbl}]</span></span>"
                    )
                chips = "".join(chips_parts)
                st.markdown(f"""
                <div class='card'>
                  <div class='lbl'>06 · Named Entities</div>
                  <div style='margin-top:.4rem'>{chips}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='card'>
                  <div class='lbl'>06 · Named Entities</div>
                  <div style='color:#8b949e;font-size:.88rem'>None detected.</div>
                </div>""", unsafe_allow_html=True)

        st.success(f"✅ Analysis complete in {elapsed*1000:.0f}ms", icon="✅")

    elif analyze and not tweet_in.strip():
        st.warning("Please enter a tweet before clicking Analyze.", icon="⚠️")

    divider()
    section("Example Tweets — click to load")

    examples = [
        ("😊 Positive", "I absolutely love this new smartphone! Best purchase I've ever made! 🎉 #happy"),
        ("😡 Negative", "Terrible customer service, waited 2 hours and got zero help. Never coming back!"),
        ("🏢 Org+GPE",  "Apple just announced a new store opening in Paris next month. Very excited!"),
        ("🏛️ Political", "The election results are in — @CNN reporting live from Washington DC. Shocking!"),
        ("🎵 Music",    "Just discovered this amazing new album by Billie Eilish. Absolute masterpiece 🎵"),
        ("😐 Neutral",  "Going to the gym later then heading to the grocery store. Normal Tuesday lol."),
    ]

    ex_cols = st.columns(3)
    for i, (label, ex) in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(f"{label}", key=f"ex_{i}", use_container_width=True):
                st.session_state["_loaded_tweet"] = ex
                st.rerun()

    if "_loaded_tweet" in st.session_state:
        st.info(f"**Loaded:** {st.session_state['_loaded_tweet']}", icon="📋")

    divider()
    section("Batch Prediction (paste multiple tweets)")

    batch_txt = st.text_area("One tweet per line", height=120,
                             placeholder="Tweet 1\nTweet 2\nTweet 3")
    if st.button("📦  Run Batch", key="batch_btn"):
        lines = [l.strip() for l in batch_txt.splitlines() if l.strip()]
        if not lines:
            st.warning("No tweets entered.")
        else:
            with st.spinner(f"Processing {len(lines)} tweets…"):
                results = []
                for line in lines:
                    cl  = clean_tweet(line)
                    s   = predict_sentiment(cl, svm_model or mlp_model, tfidf_vec)
                    results.append({"tweet": line[:80], "sentiment": s})
            bdf = pd.DataFrame(results)
            st.dataframe(bdf, use_container_width=True)

            # Download batch results
            csv_bytes = bdf.to_csv(index=False).encode()
            st.download_button("⬇️  Download batch results",
                               csv_bytes, "batch_predictions.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — TEST RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📁  Test Results":
    st.markdown("""
    <div class='ph'><h1>Test Set Results</h1>
    <p>Predicted sentiments on test.csv (unlabeled) · Model performance metrics</p></div>
    """, unsafe_allow_html=True)

    # ── Model Performance metrics ──────────────────────────────────────────
    section("Model Performance (validation split)")

    mc1, mc2 = st.columns(2)
    with mc1:
        show_img("confusion", "Confusion matrices — SVM vs MLP")
    with mc2:
        show_img("metrics", "Accuracy & F1-Score comparison")

    divider()
    section("Metrics Summary Table")

    metrics_rows = [
        ("SVM (LinearSVC)", "~0.830", "~0.830", "~15s",  "TF-IDF bigrams 30K", "✅ Best"),
        ("MLP (Deep)",      "~0.815", "~0.814", "~120s", "TF-IDF bigrams 30K", ""),
    ]
    hdr = "<tr>" + "".join(f"<th>{h}</th>" for h in
          ["Model","Accuracy","F1 (weighted)","Train Time","Features","Note"]) + "</tr>"
    body = ""
    for row in metrics_rows:
        body += "<tr>" + "".join(
            f"<td class='g'>{v}</td>" if i in (1,2) else f"<td>{v}</td>"
            for i,v in enumerate(row)
        ) + "</tr>"
    st.markdown(f"""
    <div class='card' style='padding:0'>
      <table class='mtbl'><thead>{hdr}</thead><tbody>{body}</tbody></table>
    </div>""", unsafe_allow_html=True)

    divider()
    section("Predicted Sentiments — test.csv")

    if not pred_df.empty and "predicted_sentiment" in pred_df.columns:
        # KPIs
        counts = pred_df["predicted_sentiment"].value_counts()
        kp1, kp2, kp3 = st.columns(3)
        kpi(kp1, f"{len(pred_df):,}",
            "Tweets Predicted")
        kpi(kp2, f"{counts.get('positive',0):,}",
            "😊 Predicted Positive")
        kpi(kp3, f"{counts.get('negative',0):,}",
            "😡 Predicted Negative")

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        # Distribution plot (live)
        with st.spinner("Plotting distribution…"):
            st.image(plot_pred_dist(pred_df), use_container_width=True)

        divider()
        section("Prediction Sample")
        st.dataframe(
            pred_df.head(50).reset_index(drop=True),
            use_container_width=True, height=360,
        )

        divider()
        section("Filter & Explore")
        fil = st.selectbox("Show", ["All", "Positive only", "Negative only"])
        filtered = pred_df.copy()
        if fil == "Positive only":
            filtered = filtered[filtered["predicted_sentiment"]=="positive"]
        elif fil == "Negative only":
            filtered = filtered[filtered["predicted_sentiment"]=="negative"]

        st.dataframe(filtered.reset_index(drop=True),
                     use_container_width=True, height=300)

        divider()
        section("Download")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_all = pred_df.to_csv(index=False).encode()
            st.download_button("⬇️  Download all predictions (CSV)",
                               csv_all, "test_predictions.csv", "text/csv",
                               use_container_width=True)
        with col_dl2:
            csv_pos = filtered.to_csv(index=False).encode()
            st.download_button(f"⬇️  Download filtered ({fil})",
                               csv_pos, "filtered_predictions.csv", "text/csv",
                               use_container_width=True)

    else:
        st.info("Predictions not found. Run `code.py` to generate `outputs/test_predictions.csv`.")
        show_img("pred_dist", "Prediction distribution (precomputed)")

        # Offer to run predictions live
        if not test_df.empty:
            divider()
            section("Generate Predictions Now")
            if st.button("🤖  Predict on test.csv (live)", use_container_width=False):
                active = svm_model or mlp_model
                if active is None:
                    st.error("No classifier found. Run code.py first.")
                else:
                    with st.spinner(f"Predicting {len(test_df):,} tweets…"):
                        preds = []
                        col_t = "clean_tweet" if "clean_tweet" in test_df.columns else "tweet"
                        for txt in test_df[col_t].fillna(""):
                            cl = clean_tweet(txt) if col_t == "tweet" else txt
                            preds.append(predict_sentiment(cl, active, tfidf_vec))
                        live_pred = test_df.copy()
                        live_pred["predicted_sentiment"] = preds

                    st.success(f"Done! {len(live_pred):,} tweets predicted.", icon="✅")
                    st.dataframe(live_pred.head(30), use_container_width=True)
                    csv_live = live_pred.to_csv(index=False).encode()
                    st.download_button("⬇️  Download predictions",
                                       csv_live, "test_predictions_live.csv", "text/csv")