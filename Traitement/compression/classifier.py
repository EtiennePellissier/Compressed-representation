"""
classifier.py — Classifieur de sentiment léger fondé sur la théorie de l'information
===================================================================================
 
Alternative à FinBERT (110 M de paramètres) qui garde exactement la même interface
(analyze_sentiment, analyze_sentiment_batch, compute_ticker_sentiment) pour pouvoir
être branchée dans le pipeline à la place de finbert.py.
 
Idée (cours « Machine Learning and Algorithmic Information », chapitre 4) :
    « Comprehension is compression »  (Chaitin)
 
Le modèle est construit en 4 étapes, chacune correspondant à une notion du cours :
 
1. REPRÉSENTATION COMPRESSÉE  — « hashing trick »
   Les unigrammes + bigrammes sont projetés dans 2^18 cases par une fonction de hachage
   déterministe (crc32). Pas de vocabulaire à stocker, mémoire fixe.
 
2. PONDÉRATION PAR L'ENTROPIE
   Pour chaque trait j, on mesure à quel point il « informe » sur la classe :
       w_j = KL( p(c | j)  ||  π(c) )     (en bits)
   avec p(c | j) lissé par un a priori de Dirichlet (les traits rares retombent sur π
   et reçoivent un poids ≈ 0). Les mots vides (the, of, ...) ont une distribution de
   classes ≈ π : leur poids est quasi nul, sans liste de stop-words. Seuls les
   `max_features` traits les plus informatifs sont conservés (compression avec perte).
 
3. MODÈLE À PROTOTYPES  (cours : « Prototype-based models and compression »)
   Chaque classe (positive / negative / neutral) est résumée par k prototypes
   (k-means sphérique). On jette les documents et on ne garde que les prototypes.
 
4. CHOIX DE k PAR MDL  (cours : « Minimum Description Length »)
       k* = argmin_k  [ C(M_k) + Σ_x C(x | M_k) ]
   C(M_k) = coût des coordonnées non nulles des prototypes ;
   C(x | M_k) = log2(k) + log2(1 + d(x, prototype)/δ).
 
Le score final est le même que dans finbert.py :
       sentiment_score = P(positive) − P(negative)  ∈ [−1, +1]
 
Dépendances : numpy, scipy, pandas (pas de torch, pas de transformers).
 
Utilisation en ligne de commande (depuis le dossier du fichier) :
    python classifier.py label --data-dir ../../Data/AlphaVantage --out labels.csv
    python classifier.py train --csv labels.csv --out classifier_model.npz --cutoff 20260401
    python classifier.py eval  --model classifier_model.npz --csv labels.csv --since 20260401
    python classifier.py score --model classifier_model.npz --json alphavantage_merged_20260610.json
    python classifier.py bench --model classifier_model.npz --csv labels.csv --with-finbert
"""
 
from __future__ import annotations
 
import argparse
import glob
import json
import math
import os
import re
import sys
import time
import zlib
 
import numpy as np
import pandas as pd
import scipy.sparse as sp
 
# ======================================================================
# Constantes
# ======================================================================
 
# Même ordre que dans finbert.py
LABELS = ("positive", "negative", "neutral")
_LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
 
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "classifier_model.npz"
)
 
# ======================================================================
# 1. Tokenisation
# ======================================================================
 
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*%?")
_NEGATORS = {"not", "no", "never", "without", "neither", "nor", "cannot"}
_NEG_SCOPE = 3  # nombre de mots suivant une négation qui sont « inversés »
 
 
def tokenize(text: str) -> list:
    """Découpe en mots, normalise les nombres et marque la portée des négations."""
    text = (text or "").lower().replace("’", "'")
    tokens, neg_left = [], 0
    for m in _TOKEN_RE.finditer(text):
        t = m.group()
        if t[0].isdigit():
            t = "<pct>" if t.endswith("%") else "<num>"
        elif t in _NEGATORS or t.endswith("n't"):
            neg_left = _NEG_SCOPE
        elif neg_left > 0:
            t = "not_" + t
            neg_left -= 1
        tokens.append(t)
    return tokens
 
 
def _features(tokens: list, ngram: int) -> list:
    feats = list(tokens)
    if ngram >= 2:
        feats += [a + "_" + b for a, b in zip(tokens, tokens[1:])]
    return feats
 
 
# ======================================================================
# 2. Briques « théorie de l'information »
# ======================================================================
 
def _spherical_kmeans(X: sp.csr_matrix, k: int, rng, iters: int = 25):
    """
    K-means sphérique (distance cosinus) sur une matrice creuse de lignes normalisées.
    Retourne (prototypes k×D, affectations, similarité au prototype le plus proche).
    """
    n, d = X.shape
    k = max(1, min(k, n))
    centers = np.zeros((k, d), dtype=np.float32)
 
    # Initialisation de type k-means++ (adaptée à la distance cosinus)
    centers[0] = X[int(rng.integers(n))].toarray().ravel()
    best_sim = np.asarray(X @ centers[0]).ravel()
    for c in range(1, k):
        dist = np.clip(1.0 - best_sim, 0.0, None)
        s = dist.sum()
        probs = dist / s if s > 0 else np.full(n, 1.0 / n)
        idx = int(rng.choice(n, p=probs))
        centers[c] = X[idx].toarray().ravel()
        best_sim = np.maximum(best_sim, np.asarray(X @ centers[c]).ravel())
 
    labels = None
    for _ in range(iters):
        sims = np.asarray(X @ centers.T)
        new = sims.argmax(axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            mask = labels == c
            if mask.any():
                v = np.asarray(X[mask].sum(axis=0)).ravel()
                nrm = np.linalg.norm(v)
                if nrm > 0:
                    centers[c] = v / nrm
            else:  # cluster vide : on le réinitialise sur le point le moins bien décrit
                centers[c] = X[int(sims.max(axis=1).argmin())].toarray().ravel()
 
    sims = np.asarray(X @ centers.T)
    return centers, sims.argmax(axis=1), sims.max(axis=1)
 
 
def _mdl_cost(centers: np.ndarray, best_sim: np.ndarray,
              bits_value: int = 8, delta: float = 0.02) -> float:
    """
    Longueur de description (en bits) du modèle à k prototypes :
        C(M)   = Σ coordonnées non nulles × (log2 D + bits_value)
        C(x|M) = log2 k  +  log2(1 + d(x, prototype) / δ)      pour chaque document
    """
    k, d = centers.shape
    l_model = int(np.count_nonzero(centers)) * (math.log2(max(d, 2)) + bits_value)
    resid = np.clip(1.0 - best_sim, 0.0, None)
    l_data = len(best_sim) * math.log2(k) + float(np.log2(1.0 + resid / delta).sum())
    return l_model + l_data
 
 
def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
 
 
# ======================================================================
# 3. Le classifieur
# ======================================================================
 
class EntropyPrototypeClassifier:
    """Hachage → pondération par entropie (KL) → prototypes choisis par MDL."""
 
    def __init__(self, n_features: int = 2 ** 18, max_features: int = 20000,
                 min_df: int = 2, prior_strength: float = 2.0, ngram: int = 2,
                 k_max: int = 3, val_fraction: float = 0.2, seed: int = 0):
        self.n_features = n_features
        self.max_features = max_features
        self.min_df = min_df
        self.prior_strength = prior_strength
        self.ngram = ngram
        self.k_max = k_max
        self.val_fraction = val_fraction
        self.seed = seed
        # appris par fit()
        self.kept_idx_ = None       # indices (cases de hachage) conservés
        self.weights_ = None        # poids d'entropie des traits conservés
        self.centers_ = None        # prototypes (P × D')
        self.proto_class_ = None    # classe de chaque prototype
        self.k_per_class_ = None
        self.temperature_ = 0.1
 
    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------
    def _hashed_counts(self, texts) -> sp.csr_matrix:
        indptr, indices, data = [0], [], []
        for t in texts:
            counts = {}
            for f in _features(tokenize(t if isinstance(t, str) else ""), self.ngram):
                h = zlib.crc32(f.encode("utf-8")) % self.n_features
                counts[h] = counts.get(h, 0) + 1
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))
        return sp.csr_matrix(
            (np.asarray(data, dtype=np.float32),
             np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(len(indptr) - 1, self.n_features),
        )
 
    def _fit_weights(self, Xh: sp.csr_matrix, y: np.ndarray) -> None:
        """w_j = KL( p(c|j) || π ), p(c|j) lissé par un a priori de Dirichlet."""
        n, K = Xh.shape[0], len(LABELS)
        Xb = Xh.copy()
        Xb.data[:] = 1.0                                   # présence / absence
        Y = sp.csr_matrix((np.ones(n), (y, np.arange(n))), shape=(K, n))
        counts = (Y @ Xb).toarray()                        # K × D : n_jc
        prior = np.bincount(y, minlength=K) / n            # π(c)
        df = counts.sum(axis=0)                            # n_j
        a = self.prior_strength
        p = (counts + a * prior[:, None]) / (df + a)
        kl = np.clip((p * np.log2(p / prior[:, None])).sum(axis=0), 0.0, None)
        kl[df < self.min_df] = 0.0
 
        order = np.argsort(-kl)[: self.max_features]
        order = order[kl[order] > 0]
        if order.size == 0:
            raise ValueError("Aucun trait informatif : données d'entraînement insuffisantes.")
        self.kept_idx_ = np.sort(order).astype(np.int32)
        w = kl[self.kept_idx_]
        self.weights_ = (w / w.max()).astype(np.float32)
 
    def _transform(self, Xh: sp.csr_matrix) -> sp.csr_matrix:
        X = Xh[:, self.kept_idx_].tocsr().astype(np.float32)
        X.data = np.log1p(X.data)                          # tf sous-linéaire
        X = sp.csr_matrix(X @ sp.diags(self.weights_))     # pondération entropique
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / norms) @ X, dtype=np.float32)
 
    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_labels(labels) -> np.ndarray:
        try:
            return np.array([_LABEL_TO_ID[str(l).strip().lower()] for l in labels], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"Label inconnu {e}. Labels valides : {LABELS}") from None
 
    def _fit_prototypes(self, X: sp.csr_matrix, y: np.ndarray, rng) -> None:
        centers_all, proto_class, ks = [], [], []
        for c in range(len(LABELS)):
            Xc = X[y == c]
            if Xc.shape[0] == 0:
                raise ValueError(f"Aucun exemple pour la classe '{LABELS[c]}'.")
            best = None
            for k in range(1, min(self.k_max, Xc.shape[0]) + 1):
                centers, _, best_sim = _spherical_kmeans(Xc, k, rng)
                cost = _mdl_cost(centers, best_sim)
                if best is None or cost < best[0]:
                    best = (cost, centers)
            centers_all.append(best[1])
            proto_class += [c] * best[1].shape[0]
            ks.append(int(best[1].shape[0]))
        self.centers_ = np.vstack(centers_all).astype(np.float32)
        self.proto_class_ = np.asarray(proto_class, dtype=np.int8)
        self.k_per_class_ = ks
 
    def _class_scores(self, X: sp.csr_matrix) -> np.ndarray:
        sims = np.asarray(X @ self.centers_.T)
        out = np.empty((X.shape[0], len(LABELS)), dtype=np.float32)
        for c in range(len(LABELS)):
            out[:, c] = sims[:, self.proto_class_ == c].max(axis=1)
        return out
 
    def _calibrate(self, X: sp.csr_matrix, y: np.ndarray) -> None:
        """Choisit la température du softmax qui minimise la log-loss de validation."""
        scores = self._class_scores(X)
        best_t, best_nll = self.temperature_, np.inf
        for t in np.geomspace(0.01, 1.0, 40):
            p = _softmax(scores, t)
            nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
            if nll < best_nll:
                best_t, best_nll = float(t), float(nll)
        self.temperature_ = best_t
 
    def fit(self, texts, labels) -> "EntropyPrototypeClassifier":
        texts = list(texts)
        y = self._encode_labels(labels)
        if len(texts) != len(y):
            raise ValueError("texts et labels n'ont pas la même longueur.")
        rng = np.random.default_rng(self.seed)
        n = len(texts)
        n_val = int(n * self.val_fraction) if n >= 100 else 0
        perm = rng.permutation(n)
        val, tr = perm[:n_val], perm[n_val:]
 
        Xh = self._hashed_counts(texts)
        self._fit_weights(Xh[tr], y[tr])      # poids appris sur le train uniquement
        X = self._transform(Xh)
        self._fit_prototypes(X[tr], y[tr], rng)
        if n_val > 0:
            self._calibrate(X[val], y[val])
        return self
 
    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------
    def predict_proba(self, texts) -> np.ndarray:
        """Matrice (n, 3) de probabilités, colonnes dans l'ordre LABELS."""
        if self.centers_ is None:
            raise RuntimeError("Modèle non entraîné.")
        X = self._transform(self._hashed_counts(list(texts)))
        return _softmax(self._class_scores(X), self.temperature_)
 
    def predict(self, texts) -> list:
        return [LABELS[i] for i in self.predict_proba(texts).argmax(axis=1)]
 
    def score(self, texts) -> np.ndarray:
        """P(positive) − P(negative) ∈ [−1, +1], comme dans finbert.py."""
        p = self.predict_proba(texts)
        return p[:, 0] - p[:, 1]
 
    def evaluate(self, texts, labels) -> dict:
        y = self._encode_labels(labels)
        pred = self.predict_proba(texts).argmax(axis=1)
        f1s = []
        for c in range(len(LABELS)):
            tp = int(((pred == c) & (y == c)).sum())
            fp = int(((pred == c) & (y != c)).sum())
            fn = int(((pred != c) & (y == c)).sum())
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
                "macro_f1": float(np.mean(f1s))}
 
    @property
    def n_parameters(self) -> int:
        return int(self.centers_.size + self.kept_idx_.size + self.weights_.size)
 
    # ------------------------------------------------------------------
    # Sauvegarde / chargement (npz, sans pickle)
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        if not path.endswith(".npz"):
            path += ".npz"
        meta = dict(version=1, n_features=self.n_features, ngram=self.ngram,
                    temperature=self.temperature_, k_per_class=self.k_per_class_,
                    labels=list(LABELS))
        np.savez_compressed(path, kept_idx=self.kept_idx_, weights=self.weights_,
                            centers=self.centers_, proto_class=self.proto_class_,
                            meta=np.array(json.dumps(meta)))
        return path
 
    @classmethod
    def load(cls, path: str) -> "EntropyPrototypeClassifier":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if tuple(meta["labels"]) != LABELS:
                raise ValueError("Ordre des labels incompatible avec ce module.")
            m = cls(n_features=meta["n_features"], ngram=meta["ngram"])
            m.kept_idx_ = z["kept_idx"]
            m.weights_ = z["weights"]
            m.centers_ = z["centers"]
            m.proto_class_ = z["proto_class"]
        m.temperature_ = float(meta["temperature"])
        m.k_per_class_ = list(meta["k_per_class"])
        return m
 
 
# ======================================================================
# 4. API compatible avec finbert.py (drop-in replacement)
# ======================================================================
 
_model = None
 
 
def _load_model() -> None:
    """Charge le modèle une seule fois (chemin surchargeable via CLASSIFIER_MODEL)."""
    global _model
    if _model is None:
        path = os.environ.get("CLASSIFIER_MODEL", DEFAULT_MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modèle introuvable : {path}\n"
                "Entraîne-le d'abord : python classifier.py label ... puis python classifier.py train ..."
            )
        _model = EntropyPrototypeClassifier.load(path)
 
 
def _to_dicts(probs: np.ndarray) -> list:
    return [{lab: round(float(p[i]), 4) for i, lab in enumerate(LABELS)} for p in probs]
 
 
def analyze_sentiment(text: str) -> dict:
    """{"positive": 0.7, "negative": 0.1, "neutral": 0.2} — même format que FinBERT."""
    _load_model()
    return _to_dicts(_model.predict_proba([text]))[0]
 
 
def analyze_sentiment_batch(texts: list, batch_size: int = 512) -> list:
    """Liste de dicts de probabilités, dans le même ordre que `texts`."""
    _load_model()
    out = []
    for i in range(0, len(texts), batch_size):
        out += _to_dicts(_model.predict_proba(texts[i: i + batch_size]))
    return out
 
 
def compute_ticker_sentiment(articles: dict) -> pd.DataFrame:
    """
    articles : {"NKE": ["texte1", ...], "TGT": [...]}
    Retourne un DataFrame : ticker | sentiment_score | n_articles
    (identique à finbert.compute_ticker_sentiment)
    """
    _load_model()
    rows = []
    for ticker, texts in articles.items():
        if not texts:
            print(f"[Classifier] Aucun article pour {ticker}, score = 0.0")
            rows.append({"ticker": ticker, "sentiment_score": 0.0, "n_articles": 0})
            continue
        mean_score = float(_model.score(texts).mean())
        rows.append({"ticker": ticker, "sentiment_score": round(mean_score, 4),
                     "n_articles": len(texts)})
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
 
 
# ======================================================================
# 5. Données : corpus Alpha Vantage + étiquetage par FinBERT (distillation)
# ======================================================================
 
def load_news_dir(data_dir: str) -> pd.DataFrame:
    """Lit tous les alphavantage_merged_YYYYMMDD.json → DataFrame date|ticker|text."""
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "alphavantage_merged_*.json"))):
        m = re.search(r"(\d{8})", os.path.basename(path))
        if not m:
            continue
        with open(path, "r", encoding="utf-8") as f:
            day = json.load(f)
        for ticker, texts in day.items():
            rows += [{"date": int(m.group(1)), "ticker": ticker, "text": t}
                     for t in texts if isinstance(t, str) and t.strip()]
    return pd.DataFrame(rows, columns=["date", "ticker", "text"])
 
 
def _get_teacher():
    """Importe FinBERT (le « professeur ») uniquement quand on en a besoin."""
    try:
        from finbert import analyze_sentiment_batch as teacher
    except ImportError:
        from Traitement.sentiment.finbert import analyze_sentiment_batch as teacher
    return teacher
 
 
def label_with_teacher(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """Étiquette chaque texte avec FinBERT (à faire une seule fois, puis on met en cache)."""
    teacher = _get_teacher()
    res = teacher(df["text"].tolist(), batch_size=batch_size)
    out = df.copy()
    out["p_pos"] = [r["positive"] for r in res]
    out["p_neg"] = [r["negative"] for r in res]
    out["p_neu"] = [r["neutral"] for r in res]
    out["label"] = [max(r, key=r.get) for r in res]
    return out
 
 
# ======================================================================
# 6. Ligne de commande
# ======================================================================
 
def _report(model, df, title):
    ev = model.evaluate(df["text"].tolist(), df["label"].tolist())
    print(f"[{title}] n={ev['n']}  accuracy={ev['accuracy']:.3f}  macro-F1={ev['macro_f1']:.3f}")
    if {"p_pos", "p_neg"} <= set(df.columns):
        ours = pd.Series(model.score(df["text"].tolist()))
        teacher = (df["p_pos"] - df["p_neg"]).reset_index(drop=True)
        print(f"[{title}] corrélation de Spearman avec le score FinBERT : "
              f"{ours.corr(teacher, method='spearman'):.3f}")
 
 
def cmd_label(a):
    df = load_news_dir(a.data_dir)
    if df.empty:
        sys.exit(f"Aucun article trouvé dans {a.data_dir}")
    print(f"{len(df)} articles à étiqueter avec FinBERT...")
    label_with_teacher(df, a.batch_size).to_csv(a.out, index=False)
    print(f"Étiquettes sauvegardées dans {a.out}")
 
 
def cmd_train(a):
    df = pd.read_csv(a.csv).dropna(subset=["text", "label"])
    if a.cutoff and "date" in df.columns:
        train, test = df[df["date"] < a.cutoff], df[df["date"] >= a.cutoff]
    elif a.final:
        train, test = df, None
    else:
        test = df.sample(frac=0.2, random_state=0)
        train = df.drop(test.index)
    model = EntropyPrototypeClassifier(max_features=a.max_features, k_max=a.k_max)
    t0 = time.perf_counter()
    model.fit(train["text"].tolist(), train["label"].tolist())
    print(f"Entraînement : {time.perf_counter() - t0:.1f}s sur {len(train)} textes ; "
          f"prototypes par classe {dict(zip(LABELS, model.k_per_class_))} ; "
          f"{len(model.kept_idx_)} traits conservés ; T={model.temperature_:.3f}")
    path = model.save(a.out)
    print(f"Modèle : {path} ({os.path.getsize(path) / 1024:.0f} Ko, {model.n_parameters:,} paramètres)")
    if test is not None and len(test):
        _report(model, test, "test")
 
 
def cmd_eval(a):
    model = EntropyPrototypeClassifier.load(a.model)
    df = pd.read_csv(a.csv).dropna(subset=["text", "label"])
    if a.since and "date" in df.columns:
        df = df[df["date"] >= a.since]
    _report(model, df, "eval")
 
 
def cmd_score(a):
    os.environ["CLASSIFIER_MODEL"] = a.model
    with open(a.json, "r", encoding="utf-8") as f:
        articles = json.load(f)
    df = compute_ticker_sentiment(articles)
    print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)
 
 
def cmd_bench(a):
    model = EntropyPrototypeClassifier.load(a.model)
    texts = pd.read_csv(a.csv)["text"].astype(str).tolist()[: a.n]
    t0 = time.perf_counter()
    model.predict_proba(texts)
    dt = time.perf_counter() - t0
    print(f"Classifieur : {len(texts) / dt:,.0f} textes/s ; "
          f"{os.path.getsize(a.model) / 1024:.0f} Ko sur disque ; {model.n_parameters:,} paramètres")
    if a.with_finbert:
        teacher = _get_teacher()
        t0 = time.perf_counter()
        teacher(texts[: min(len(texts), 100)])
        dt2 = time.perf_counter() - t0
        print(f"FinBERT     : {min(len(texts), 100) / dt2:,.1f} textes/s ; ≈110 000 000 paramètres")
 
 
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
 
    s = sub.add_parser("label", help="étiquette le corpus Alpha Vantage avec FinBERT (une seule fois)")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--out", default="labels.csv")
    s.add_argument("--batch-size", type=int, default=16)
    s.set_defaults(fn=cmd_label)
 
    s = sub.add_parser("train", help="entraîne le classifieur")
    s.add_argument("--csv", required=True, help="colonnes : text, label [, date, p_pos, p_neg]")
    s.add_argument("--out", default=DEFAULT_MODEL_PATH)
    s.add_argument("--cutoff", type=int, help="date YYYYMMDD : train avant, test à partir de")
    s.add_argument("--final", action="store_true", help="entraîne sur 100%% des données")
    s.add_argument("--max-features", type=int, default=20000)
    s.add_argument("--k-max", type=int, default=3)
    s.set_defaults(fn=cmd_train)
 
    s = sub.add_parser("eval", help="évalue un modèle")
    s.add_argument("--model", default=DEFAULT_MODEL_PATH)
    s.add_argument("--csv", required=True)
    s.add_argument("--since", type=int)
    s.set_defaults(fn=cmd_eval)
 
    s = sub.add_parser("score", help="score un fichier JSON quotidien par ticker")
    s.add_argument("--model", default=DEFAULT_MODEL_PATH)
    s.add_argument("--json", required=True)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_score)
 
    s = sub.add_parser("bench", help="vitesse et taille (et comparaison avec FinBERT)")
    s.add_argument("--model", default=DEFAULT_MODEL_PATH)
    s.add_argument("--csv", required=True)
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--with-finbert", action="store_true")
    s.set_defaults(fn=cmd_bench)
 
    args = p.parse_args()
    args.fn(args)
 
 
if __name__ == "__main__":
    main()
