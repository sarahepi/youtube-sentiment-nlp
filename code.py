"""
=============================================================================
MINI-PROJET : Analyse des Médias et Réseaux Sociaux  —  SG01
Dataset     : 1.6 Million Tweets (Kaggle)

  train.csv  →  3 colonnes : index | sentiment | tweet
  test.csv   →  2 colonnes : index | tweet          (pas de labels)

Pipeline :
  Phase 1 → EDA            (statistiques, visualisations, WordCloud brut)
  Phase 2 → Ingénierie     (NLTK nettoyage, spaCy lemma+POS, NER, WordCloud propre)
  Phase 3 → Modélisation   (LDA, Word2Vec, SVM vs MLP + prédictions sur test)
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & CONFIGURATION GLOBALE
# ──────────────────────────────────────────────────────────────────────────────
import os, re, time, warnings, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

import spacy
from wordcloud import WordCloud

from gensim import corpora
from gensim.models import LdaModel, Word2Vec
from gensim.models.coherencemodel import CoherenceModel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Téléchargements NLTK ─────────────────────────────────────────────────────
for _pkg in ["punkt", "stopwords", "averaged_perceptron_tagger", "wordnet"]:
    nltk.download(_pkg, quiet=True)

# ── Modèle spaCy ─────────────────────────────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    log.warning("Modèle spaCy absent — exécutez : python -m spacy download en_core_web_sm")
    NLP = None

# ── Chemins ───────────────────────────────────────────────────────────────────
TRAIN_PATH = "train.csv"   # colonnes : index | sentiment | tweet
TEST_PATH  = "test.csv"    # colonnes : index | tweet        (sans labels)

# ── Paramètres ────────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 50_000   # sous-échantillon pour les passes coûteuses (LDA, W2V, NER)
N_TOPICS     = 6        # nombre de thèmes LDA
RANDOM_STATE = 42
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

# ── Stopwords personnalisés (domaine Twitter) ─────────────────────────────────
CUSTOM_STOPWORDS = set(stopwords.words("english")) | {
    "rt", "amp", "http", "https", "www", "com", "co",
    "via", "im", "ur", "u", "r", "got", "get", "go",
    "like", "just", "dont", "cant", "know", "think",
    "would", "could", "really", "still", "even", "much",
    "want", "make", "day", "time", "today", "said",
}

# ── Mapping des labels de sentiment ───────────────────────────────────────────
SENTIMENT_MAP = {
    0: "negative", 4: "positive",
    1: "positive", -1: "negative",
    "0": "negative", "4": "positive",
    "1": "positive", "-1": "negative",
    "neg": "negative", "pos": "positive",
    "negative": "negative", "positive": "positive",
}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(train_path: str, test_path: str):
    """
    Charge train.csv  (index | sentiment | tweet)
          et test.csv (index | tweet)  — sans colonne sentiment.

    Retourne : train_df (avec labels), test_df (sans labels).
    """
    # ── train ─────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(train_path, encoding="utf-8", on_bad_lines="skip")

    # Normaliser le nom de la colonne texte si nécessaire
    for c in train_df.columns:
        if c.lower() in ("tweet", "text", "tweets", "texts") and c != "tweet":
            train_df.rename(columns={c: "tweet"}, inplace=True)
            break

    if "tweet" not in train_df.columns:
        raise ValueError(f"Colonne texte introuvable dans train.csv. "
                         f"Colonnes : {list(train_df.columns)}")
    if "sentiment" not in train_df.columns:
        raise ValueError(f"Colonne 'sentiment' introuvable dans train.csv. "
                         f"Colonnes : {list(train_df.columns)}")

    # Normaliser les labels
    train_df["sentiment"] = train_df["sentiment"].map(
        lambda x: SENTIMENT_MAP.get(x, SENTIMENT_MAP.get(str(x), None))
    )
    train_df = train_df[train_df["sentiment"].isin(["negative", "positive"])].reset_index(drop=True)

    # ── test ──────────────────────────────────────────────────────────────────
    test_df = pd.read_csv(test_path, encoding="utf-8", on_bad_lines="skip")

    for c in test_df.columns:
        if c.lower() in ("tweet", "text", "tweets", "texts") and c != "tweet":
            test_df.rename(columns={c: "tweet"}, inplace=True)
            break

    if "tweet" not in test_df.columns:
        raise ValueError(f"Colonne texte introuvable dans test.csv. "
                         f"Colonnes : {list(test_df.columns)}")

    # Confirmer qu'il n'y a PAS de colonne sentiment dans test
    if "sentiment" in test_df.columns:
        log.warning("test.csv contient une colonne 'sentiment' — elle sera ignorée pour l'évaluation.")
        test_df = test_df.drop(columns=["sentiment"])

    log.info(f"train.csv : {len(train_df):,} lignes  (avec labels)")
    log.info(f"test.csv  : {len(test_df):,} lignes  (sans labels — prédiction uniquement)")
    return train_df, test_df


def eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Analyse exploratoire : stats, distributions, longueurs."""
    log.info("── Phase 1 : EDA ──────────────────────────────────────────────")

    # ── Résumé console ────────────────────────────────────────────────────────
    print("\n📊  TRAIN :")
    print(f"  Shape     : {train_df.shape}")
    print(f"  Colonnes  : {list(train_df.columns)}")
    print(f"  Doublons  : {train_df.duplicated(subset='tweet').sum():,}")
    print(f"  NaN tweet : {train_df['tweet'].isna().sum():,}")
    print(f"  Sentiment :\n{train_df['sentiment'].value_counts().to_string()}")

    print("\n📊  TEST (sans labels) :")
    print(f"  Shape     : {test_df.shape}")
    print(f"  Colonnes  : {list(test_df.columns)}")
    print(f"  Doublons  : {test_df.duplicated(subset='tweet').sum():,}")
    print(f"  NaN tweet : {test_df['tweet'].isna().sum():,}")

    # Métriques de longueur (uniquement sur train car test n'a pas de labels)
    df = train_df.copy()
    df["text_len"]   = df["tweet"].astype(str).apply(len)
    df["word_count"] = df["tweet"].astype(str).apply(lambda x: len(x.split()))

    print("\n📐  Statistiques des longueurs — TRAIN :")
    print(df[["text_len", "word_count"]].describe().round(2))

    # ── Figure 1 : distributions train ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 1 — EDA : Analyse du Dataset Twitter (train)",
                 fontsize=14, fontweight="bold")

    # Distribution sentiment
    counts = df["sentiment"].value_counts()
    bars = axes[0].bar(counts.index, counts.values,
                       color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
    axes[0].set_title("Distribution des Sentiments")
    axes[0].set_ylabel("Nombre de tweets")
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + counts.max() * 0.01,
                     f"{v:,}", ha="center", va="bottom", fontsize=10)

    # Histogramme longueurs
    axes[1].hist(df["text_len"], bins=60, color="#3498db", edgecolor="white", alpha=0.85)
    axes[1].set_title("Distribution des Longueurs (caractères)")
    axes[1].set_xlabel("Longueur")
    axes[1].set_ylabel("Fréquence")

    # Boxplot mots par sentiment
    df.boxplot(column="word_count", by="sentiment", ax=axes[2],
               patch_artist=True, boxprops=dict(facecolor="#9b59b6", alpha=0.6))
    axes[2].set_title("Nombre de mots par Sentiment")
    axes[2].set_xlabel("Sentiment")
    axes[2].set_ylabel("Mots")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_distributions.png", bbox_inches="tight")
    plt.show()

    # ── Figure 2 : train sentiment pie  +  comparaison tailles train/test ─────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 1 — Aperçu Train vs Test", fontsize=13, fontweight="bold")

    # Pie sentiment sur train
    c = train_df["sentiment"].value_counts()
    axes[0].pie(c.values, labels=c.index, autopct="%1.1f%%",
                colors=["#e74c3c", "#2ecc71"], startangle=90,
                wedgeprops=dict(edgecolor="white"))
    axes[0].set_title(f"Sentiments — Train ({len(train_df):,} tweets)")

    # Bar comparaison tailles
    sizes = {"train\n(avec labels)": len(train_df), "test\n(sans labels)": len(test_df)}
    axes[1].bar(sizes.keys(), sizes.values(),
                color=["#2980b9", "#f39c12"], edgecolor="white", width=0.4)
    axes[1].set_title("Taille Train vs Test")
    axes[1].set_ylabel("Nombre de tweets")
    for bar, v in zip(axes[1].patches, sizes.values()):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(sizes.values()) * 0.01,
                     f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_train_vs_test.png", bbox_inches="tight")
    plt.show()
    log.info("✅  Figures EDA sauvegardées")
    return df   # train_df enrichi des colonnes text_len / word_count


def wordcloud_raw(train_df: pd.DataFrame) -> None:
    """WordCloud brut avant nettoyage (sur train uniquement)."""
    log.info("Génération WordCloud brut (train) …")
    n = min(20_000, len(train_df))
    text = " ".join(train_df["tweet"].dropna().sample(n, random_state=RANDOM_STATE))
    wc = WordCloud(width=1200, height=600, background_color="black",
                   colormap="plasma", max_words=200,
                   stopwords=CUSTOM_STOPWORDS, collocations=False).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud — Termes Fréquents AVANT Nettoyage (train)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/wordcloud_raw.png", bbox_inches="tight")
    plt.show()
    log.info("✅  Figure sauvegardée : wordcloud_raw.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — INGÉNIERIE DU TEXTE
# ══════════════════════════════════════════════════════════════════════════════

# ── 2.1 Nettoyage NLTK ───────────────────────────────────────────────────────

def _clean_one(text: str, tokenizer: TweetTokenizer) -> str:
    """Nettoie un tweet : URLs, mentions, hashtags, ponctuation, stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # URLs
    text = re.sub(r"@\w+", "", text)                 # mentions
    text = re.sub(r"#(\w+)", r"\1", text)            # hashtags → mot
    text = re.sub(r"[^a-zA-Z\s']", " ", text)       # ponctuation / chiffres
    text = re.sub(r"\s+", " ", text).strip()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def apply_nltk_cleaning(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Applique le nettoyage NLTK sur la colonne 'tweet' → 'clean_tweet'."""
    log.info(f"NLTK — Nettoyage {label} ({len(df):,} tweets) …")
    tk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    t0 = time.time()
    df = df.copy()
    df["clean_tweet"] = df["tweet"].apply(lambda x: _clean_one(x, tk))
    log.info(f"  → Terminé en {time.time() - t0:.1f}s")
    return df


# ── 2.2 Lemmatisation & POS Tagging spaCy ────────────────────────────────────

def _lemmatize_one(text: str) -> str:
    """Lemmatise un texte et conserve NOUN / VERB / ADJ / ADV."""
    if NLP is None:
        return text
    allowed = {"NOUN", "VERB", "ADJ", "ADV"}
    doc = NLP(text, disable=["parser", "ner"])
    return " ".join(
        tok.lemma_ for tok in doc
        if tok.pos_ in allowed and not tok.is_stop and len(tok.lemma_) > 2
    )


def apply_spacy_lemma(df: pd.DataFrame, sample_n: int = SAMPLE_SIZE,
                      label: str = "") -> pd.DataFrame:
    """
    Lemmatise un sous-échantillon (trop coûteux sur 1.6M).
    Les lignes non sélectionnées conservent leur clean_tweet.
    """
    n = min(sample_n, len(df))
    log.info(f"spaCy — Lemmatisation {label} sur {n:,} tweets …")
    df = df.copy()
    df["lemma_tweet"] = df["clean_tweet"].copy()

    if NLP is None:
        return df

    idx = df.sample(n, random_state=RANDOM_STATE).index
    t0 = time.time()
    df.loc[idx, "lemma_tweet"] = [_lemmatize_one(t) for t in df.loc[idx, "clean_tweet"]]
    log.info(f"  → Terminé en {time.time() - t0:.1f}s")
    return df


# ── 2.3 NER spaCy ────────────────────────────────────────────────────────────

def extract_ner(train_df: pd.DataFrame, sample_n: int = 5_000) -> None:
    """
    Extraction des entités nommées sur le TRAIN uniquement
    (PERSON, ORG, GPE, LOC, NORP) — visualisation + top entités par type.
    """
    log.info(f"spaCy NER — extraction sur {min(sample_n, len(train_df)):,} tweets (train) …")
    if NLP is None:
        log.warning("spaCy non disponible, NER ignoré.")
        return

    sample = (train_df["tweet"].dropna()
              .sample(min(sample_n, len(train_df)), random_state=RANDOM_STATE))
    entities = []
    for doc in NLP.pipe(sample, batch_size=256, disable=["tagger", "parser"]):
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "NORP"}:
                entities.append({"entity": ent.text.strip(), "label": ent.label_})

    if not entities:
        log.warning("Aucune entité détectée.")
        return

    ner_df = pd.DataFrame(entities)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 2 — Extraction d'Entités Nommées (NER) — train",
                 fontsize=13, fontweight="bold")

    top_ents = ner_df["entity"].value_counts().head(20)
    axes[0].barh(top_ents.index[::-1], top_ents.values[::-1], color="#e67e22")
    axes[0].set_title("Top 20 Entités Détectées")
    axes[0].set_xlabel("Fréquence")

    label_counts = ner_df["label"].value_counts()
    axes[1].pie(label_counts.values, labels=label_counts.index,
                autopct="%1.1f%%", colors=sns.color_palette("Set2"),
                wedgeprops=dict(edgecolor="white"))
    axes[1].set_title("Répartition par Type d'Entité")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ner_analysis.png", bbox_inches="tight")
    plt.show()
    log.info("✅  Figure sauvegardée : ner_analysis.png")

    print("\n🏷️  Top entités par type :")
    for lbl in ner_df["label"].unique():
        top = ner_df[ner_df["label"] == lbl]["entity"].value_counts().head(5)
        print(f"  [{lbl}]  {list(top.index)}")


def wordcloud_clean(train_df: pd.DataFrame) -> None:
    """WordCloud après nettoyage + lemmatisation (train)."""
    col = "lemma_tweet" if "lemma_tweet" in train_df.columns else "clean_tweet"
    log.info(f"Génération WordCloud propre (colonne '{col}') …")
    n = min(20_000, len(train_df))
    text = " ".join(train_df[col].dropna().sample(n, random_state=RANDOM_STATE))
    wc = WordCloud(width=1200, height=600, background_color="white",
                   colormap="viridis", max_words=200, collocations=False).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud — Termes Fréquents APRÈS Nettoyage & Lemmatisation (train)",
              fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/wordcloud_clean.png", bbox_inches="tight")
    plt.show()
    log.info("✅  Figure sauvegardée : wordcloud_clean.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MODÉLISATION AVANCÉE
# ══════════════════════════════════════════════════════════════════════════════

# ── 3.1 Topic Modeling LDA (Gensim) ──────────────────────────────────────────

def train_lda(train_df: pd.DataFrame, sample_n: int = SAMPLE_SIZE) -> LdaModel:
    """Entraîne LDA sur le train et affiche les thèmes latents."""
    n = min(sample_n, len(train_df))
    log.info(f"Gensim LDA — entraînement sur {n:,} tweets (train) …")
    col = "lemma_tweet" if "lemma_tweet" in train_df.columns else "clean_tweet"

    corpus_texts = (
        train_df[col].dropna()
          .sample(n, random_state=RANDOM_STATE)
          .apply(str.split).tolist()
    )

    dictionary = corpora.Dictionary(corpus_texts)
    dictionary.filter_extremes(no_below=10, no_above=0.4, keep_n=10_000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_texts]

    lda_model = LdaModel(
        corpus=bow_corpus, id2word=dictionary,
        num_topics=N_TOPICS, random_state=RANDOM_STATE,
        passes=5, alpha="auto", per_word_topics=True,
    )

    coherence = CoherenceModel(
        model=lda_model, texts=corpus_texts,
        dictionary=dictionary, coherence="c_v"
    ).get_coherence()
    log.info(f"  → Cohérence LDA (c_v) : {coherence:.4f}")

    print(f"\n📌  Thèmes LDA (n={N_TOPICS}) — Cohérence c_v = {coherence:.4f}")
    for idx, topic in lda_model.print_topics(num_words=8):
        print(f"  Thème {idx + 1} : {topic}")

    # Visualisation des poids par thème
    topic_words, weights = [], []
    for t_id in range(N_TOPICS):
        pairs = lda_model.show_topic(t_id, topn=6)
        topic_words.append([w for w, _ in pairs])
        weights.append([round(s, 3) for _, s in pairs])

    rows = (N_TOPICS + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4 + 1))
    fig.suptitle(f"Phase 3 — LDA Topic Modeling ({N_TOPICS} thèmes)",
                 fontsize=13, fontweight="bold")
    palette = sns.color_palette("tab10", N_TOPICS)
    for i, ax in enumerate(axes.flat):
        if i >= N_TOPICS:
            ax.axis("off"); continue
        ax.barh(topic_words[i][::-1], weights[i][::-1], color=palette[i])
        ax.set_title(f"Thème {i + 1}", fontweight="bold")
        ax.set_xlabel("Poids")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lda_topics.png", bbox_inches="tight")
    plt.show()
    log.info("✅  Figure sauvegardée : lda_topics.png")
    return lda_model


# ── 3.2 Word2Vec (Gensim) ────────────────────────────────────────────────────

def train_word2vec(train_df: pd.DataFrame, sample_n: int = SAMPLE_SIZE) -> Word2Vec:
    """Entraîne Word2Vec Skip-gram sur le train et visualise les similarités."""
    n = min(sample_n, len(train_df))
    log.info(f"Gensim Word2Vec — entraînement sur {n:,} tweets (train) …")
    col = "lemma_tweet" if "lemma_tweet" in train_df.columns else "clean_tweet"

    sentences = (
        train_df[col].dropna()
          .sample(n, random_state=RANDOM_STATE)
          .apply(str.split).tolist()
    )

    w2v = Word2Vec(
        sentences=sentences, vector_size=100, window=5,
        min_count=5, workers=4, sg=1, epochs=5, seed=RANDOM_STATE,
    )
    log.info(f"  → Vocabulaire Word2Vec : {len(w2v.wv):,} mots")

    probe_words = ["twitter", "love", "hate", "news", "music", "happy", "sad", "work"]
    print("\n🔗  Mots sémantiquement proches (Word2Vec) :")
    vocab_probe = []
    for word in probe_words:
        if word in w2v.wv:
            vocab_probe.append(word)
            similar = w2v.wv.most_similar(word, topn=5)
            print(f"  '{word}' → {[w for w, _ in similar]}")

    if len(vocab_probe) >= 2:
        sim_matrix = np.array([
            [w2v.wv.similarity(a, b) for b in vocab_probe]
            for a in vocab_probe
        ])
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=vocab_probe, yticklabels=vocab_probe, linewidths=0.5)
        plt.title("Phase 3 — Similarités Cosinus Word2Vec", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/word2vec_similarity.png", bbox_inches="tight")
        plt.show()
        log.info("✅  Figure sauvegardée : word2vec_similarity.png")

    return w2v


# ── 3.3 Classification SVM vs MLP ────────────────────────────────────────────

def run_classification(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       sample_n: int = 100_000) -> dict:
    """
    Entraînement + évaluation sur un split interne du TRAIN (80/20).
    Puis prédiction sur TEST (pas de labels disponibles).
    Sauvegarde les prédictions dans outputs/test_predictions.csv.
    """
    log.info("Classification SVM vs MLP …")
    col = "lemma_tweet" if "lemma_tweet" in train_df.columns else "clean_tweet"

    # ── Préparer le train ─────────────────────────────────────────────────────
    sub = (train_df[["sentiment", col]].dropna()
             .pipe(lambda d: d[d["sentiment"].isin(["positive", "negative"])]))
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=RANDOM_STATE)

    # Split interne 80/20 pour l'évaluation (test.csv n'a pas de labels)
    train_part, val_part = train_test_split(
        sub, test_size=0.2, random_state=RANDOM_STATE, stratify=sub["sentiment"]
    )
    log.info(f"  Entraînement interne : {len(train_part):,}  |  Validation : {len(val_part):,}")

    le = LabelEncoder()
    le.fit(["negative", "positive"])
    y_train = le.transform(train_part["sentiment"])
    y_val   = le.transform(val_part["sentiment"])

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr  = tfidf.fit_transform(train_part[col])
    X_val = tfidf.transform(val_part[col])

    # Préparer test.csv pour la prédiction finale
    test_col = "lemma_tweet" if "lemma_tweet" in test_df.columns else "clean_tweet"
    if test_col not in test_df.columns:
        log.warning("test_df n'a pas encore été nettoyé — colonne clean_tweet manquante.")
        X_test_pred = None
    else:
        X_test_pred = tfidf.transform(test_df[test_col].fillna(""))

    # ── Modèles ───────────────────────────────────────────────────────────────
    models = {
        "SVM (LinearSVC)": LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        "MLP (Deep)"     : MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            max_iter=50, learning_rate_init=1e-3,
            early_stopping=True, n_iter_no_change=5,
            random_state=RANDOM_STATE, verbose=False
        ),
    }

    results = {}
    best_model_name, best_f1, best_clf = None, -1, None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 3 — SVM vs MLP : Analyse de Sentiment (validation interne)",
                 fontsize=13, fontweight="bold")

    for ax, (name, clf) in zip(axes, models.items()):
        log.info(f"  Entraînement : {name} …")
        t0 = time.time()
        clf.fit(X_tr, y_train)
        elapsed = time.time() - t0

        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="weighted")
        results[name] = {"accuracy": acc, "f1_weighted": f1, "time_s": round(elapsed, 1)}
        log.info(f"  {name} → Accuracy={acc:.4f}  F1={f1:.4f}  ({elapsed:.1f}s)")

        print(f"\n📋  {name} — Rapport de classification (validation) :")
        print(classification_report(y_val, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_val, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAcc={acc:.3f}  F1={f1:.3f}", fontweight="bold")

        if f1 > best_f1:
            best_f1, best_model_name, best_clf = f1, name, clf

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/classification_confusion.png", bbox_inches="tight")
    plt.show()

    # Bar chart métriques
    recap = pd.DataFrame(results).T
    recap[["accuracy", "f1_weighted"]].plot(
        kind="bar", figsize=(8, 5),
        color=["#2980b9", "#27ae60"], edgecolor="white", width=0.5
    )
    plt.title("Comparaison Accuracy & F1-Score : SVM vs MLP", fontsize=12, fontweight="bold")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.ylim(0.7, 1.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", bbox_inches="tight")
    plt.show()

    print("\n📊  Récapitulatif métriques :")
    print(recap.round(4).to_string())

    # ── Prédictions sur test.csv (sans labels) ────────────────────────────────
    if X_test_pred is not None and best_clf is not None:
        log.info(f"  Prédiction sur test.csv avec le meilleur modèle : {best_model_name} …")
        preds = best_clf.predict(X_test_pred)
        pred_labels = le.inverse_transform(preds)

        pred_df = test_df[["index", "tweet"]].copy() if "index" in test_df.columns \
                  else test_df[["tweet"]].copy()
        pred_df["predicted_sentiment"] = pred_labels

        out_path = f"{OUTPUT_DIR}/test_predictions.csv"
        pred_df.to_csv(out_path, index=False)
        log.info(f"✅  Prédictions sauvegardées : {out_path}")

        # Distribution des prédictions sur test
        pred_counts = pred_df["predicted_sentiment"].value_counts()
        plt.figure(figsize=(6, 4))
        plt.bar(pred_counts.index, pred_counts.values,
                color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.4)
        plt.title(f"Distribution des Prédictions — test.csv\n(modèle : {best_model_name})",
                  fontsize=12, fontweight="bold")
        plt.ylabel("Nombre de tweets")
        for bar, v in zip(plt.gca().patches, pred_counts.values):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + pred_counts.max() * 0.01,
                     f"{v:,}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/test_predictions_distribution.png", bbox_inches="tight")
        plt.show()
        log.info("✅  Figure sauvegardée : test_predictions_distribution.png")

        print(f"\n🔮  Prédictions sur test.csv ({len(pred_df):,} tweets) :")
        print(pred_counts.to_string())

    log.info("✅  Figures classification sauvegardées")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Orchestrateur du pipeline complet
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MINI-PROJET NLP — Analyse des Médias et Réseaux Sociaux (SG01)")
    print("  train.csv : index | sentiment | tweet")
    print("  test.csv  : index | tweet          (sans labels → prédiction)")
    print("=" * 70)

    # Vérification des fichiers
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fichier introuvable : '{path}'\n"
                "→ Placez train.csv et test.csv dans le même dossier que ce script."
            )

    # ── PHASE 1 : EDA ─────────────────────────────────────────────────────────
    train_df, test_df = load_dataset(TRAIN_PATH, TEST_PATH)
    train_df = eda(train_df, test_df)
    wordcloud_raw(train_df)

    # ── PHASE 2 : Ingénierie du Texte ─────────────────────────────────────────
    log.info("── Phase 2 : Ingénierie du Texte ──────────────────────────────")

    # Nettoyage NLTK des deux fichiers
    train_df = apply_nltk_cleaning(train_df, label="[train]")
    test_df  = apply_nltk_cleaning(test_df,  label="[test]")

    # Lemmatisation spaCy (sous-échantillon pour train, tout le test)
    train_df = apply_spacy_lemma(train_df, sample_n=SAMPLE_SIZE,       label="[train]")
    test_df  = apply_spacy_lemma(test_df,  sample_n=len(test_df),      label="[test]")

    # NER sur le train uniquement (labels disponibles)
    extract_ner(train_df, sample_n=5_000)

    # WordCloud après nettoyage
    wordcloud_clean(train_df)

    # ── PHASE 3 : Modélisation Avancée ────────────────────────────────────────
    log.info("── Phase 3 : Modélisation Avancée ─────────────────────────────")

    lda_model = train_lda(train_df, sample_n=SAMPLE_SIZE)
    w2v_model = train_word2vec(train_df, sample_n=SAMPLE_SIZE)

    # Classification : évaluation sur split interne, prédiction sur test.csv
    metrics = run_classification(train_df, test_df, sample_n=100_000)

    # Sauvegarde des modèles
    lda_model.save(f"{OUTPUT_DIR}/lda_model")
    w2v_model.save(f"{OUTPUT_DIR}/word2vec_model")
    log.info("✅  Modèles LDA et Word2Vec sauvegardés dans outputs/")

    # ── Résumé des fichiers produits ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅  Pipeline complet terminé avec succès !")
    print(f"\n  Fichiers générés dans ./{OUTPUT_DIR}/ :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}")
        print(f"    • {f:<45} {size/1024:>7.1f} Ko")
    print("=" * 70)


if __name__ == "__main__":
    main()
