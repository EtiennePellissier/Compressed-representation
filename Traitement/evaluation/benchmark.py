#!/usr/bin/env python3
"""
benchmark.py : compare des classifieurs de sentiment "légers" (théorie de
l'information) à des baselines classiques / transformers.
 
Modèles comparés
----------------
1. tfidf_logreg        : TF-IDF + régression logistique (baseline classique)
2. entropy_logreg      : features pondérées par le gain d'information (entropie),
                         top-k termes seulement + régression logistique
3. entropy_svd_logreg  : idem + compression des features par SVD tronquée
4. compress_mdl_raw    : classifieur par compression (MDL) : on choisit la classe c qui
                         minimise  C(corpus_c + x) - C(corpus_c)   (zlib)
5. compress_mdl_entropy: idem, mais le "dictionnaire" de chaque classe est construit
                         avec les termes les plus informatifs (poids d'entropie)
6. transformer (option): modèle HuggingFace pré-entraîné (ex. ProsusAI/finbert),
                         évalué en inférence seule
 
Métriques : accuracy, F1 macro, temps d'entraînement, temps d'inférence (ms / 1000
textes), taille du modèle (octets, via pickle), pic mémoire Python (tracemalloc).
 
Usage
-----
    python benchmark.py --demo
    python benchmark.py --csv data.csv --text-col text --label-col label
    python benchmark.py --hf zeroshot/twitter-financial-news-sentiment \
                        --text-col text --label-col label
    python benchmark.py --csv data.csv --transformer ProsusAI/finbert
 
Dépendances : numpy, scikit-learn  (+ pandas pour --csv, datasets pour --hf,
transformers + torch pour --transformer).
"""
from __future__ import annotations
 
import argparse
import json
import pickle
import time
import tracemalloc
import zlib
from dataclasses import dataclass, asdict
 
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
ZLIB_WINDOW = 32768  # taille max du dictionnaire / fenêtre deflate
 
 
# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
DEMO_POS = [
    "profits surge as the company beats expectations",
    "strong growth and record revenue lift the shares",
    "analysts upgrade the stock after excellent results",
    "dividend increase signals confidence and solid outlook",
    "sales jump and margins improve significantly",
]
DEMO_NEG = [
    "shares plunge after the company misses estimates",
    "weak demand and falling revenue hurt the outlook",
    "analysts downgrade the stock amid heavy losses",
    "layoffs and lawsuit weigh on profit and margins",
    "guidance cut triggers a sharp selloff",
]
DEMO_NEU = [
    "the company will hold its annual meeting on tuesday",
    "the board announced a change in the reporting schedule",
    "the firm operates stores in several european countries",
    "the report was published on the company website",
    "management will present at an industry conference",
]
 
 
def load_demo(n: int = 600, seed: int = 0):
    """Petit jeu synthétique, uniquement pour vérifier que le pipeline tourne."""
    rng = np.random.default_rng(seed)
    pools = {"positive": DEMO_POS, "negative": DEMO_NEG, "neutral": DEMO_NEU}
    texts, labels = [], []
    for _ in range(n):
        lab = rng.choice(list(pools))
        a, b = rng.choice(pools[lab], 2)
        texts.append(f"{a} . {b}")
        labels.append(lab)
    return texts, labels
 
 
def load_csv(path, text_col, label_col):
    import pandas as pd
 
    df = pd.read_csv(path).dropna(subset=[text_col, label_col])
    return df[text_col].astype(str).tolist(), df[label_col].astype(str).tolist()
 
 
def load_hf(name, text_col, label_col, config=None, split=None):
    from datasets import load_dataset, concatenate_datasets
 
    ds = load_dataset(name, config) if config else load_dataset(name)
    if split:
        d = ds[split]
    else:
        d = concatenate_datasets([ds[k] for k in ds.keys()])
    texts = [str(t) for t in d[text_col]]
    feat = d.features[label_col]
    if hasattr(feat, "names"):  # ClassLabel -> noms lisibles
        labels = [feat.names[i] for i in d[label_col]]
    else:
        labels = [str(l) for l in d[label_col]]
    return texts, labels
 
 
# ----------------------------------------------------------------------------
# Outils entropie
# ----------------------------------------------------------------------------
def entropy(p: np.ndarray, axis=-1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)
 
 
def information_gain(Xbin, y, n_classes):
    """
    Gain d'information de chaque terme t sur la classe C :
        IG(t) = H(C) - [ P(t) H(C|t) + P(~t) H(C|~t) ]
    Xbin : matrice creuse binaire (docs x termes). Retourne (IG, P(c|t)).
    """
    n_docs = Xbin.shape[0]
    Y = np.zeros((n_docs, n_classes))
    Y[np.arange(n_docs), y] = 1.0
    n_tc = np.asarray((Xbin.T @ Y))                 # (termes x classes) : docs avec t et classe c
    n_c = Y.sum(0)                                   # docs par classe
    n_t = n_tc.sum(1)                                # docs contenant t
    n_nt_c = n_c[None, :] - n_tc                     # docs sans t et classe c
    n_nt = n_docs - n_t
 
    h_c = entropy(n_c / n_docs)
    p_c_t = n_tc / np.maximum(n_t[:, None], 1)
    p_c_nt = n_nt_c / np.maximum(n_nt[:, None], 1)
    ig = h_c - (n_t / n_docs) * entropy(p_c_t) - (n_nt / n_docs) * entropy(p_c_nt)
    return np.maximum(ig, 0.0), p_c_t
 
 
# ----------------------------------------------------------------------------
# Modèles
# ----------------------------------------------------------------------------
class TfidfLogReg:
    name = "tfidf_logreg"
 
    def __init__(self, max_features=50000):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                   max_features=max_features)
        self.clf = LogisticRegression(max_iter=1000)
 
    def fit(self, X, y):
        self.clf.fit(self.vec.fit_transform(X), y)
        return self
 
    def predict(self, X):
        return self.clf.predict(self.vec.transform(X))
 
 
class EntropyLogReg:
    """TF-IDF pondéré par IG(t), uniquement les top-k termes les plus informatifs."""
    name = "entropy_logreg"
 
    def __init__(self, top_k=2000, svd_dim=None):
        self.top_k, self.svd_dim = top_k, svd_dim
        if svd_dim:
            self.name = "entropy_svd_logreg"
        self.cv = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True)
        self.tfidf = None
        self.svd = None
        self.clf = LogisticRegression(max_iter=1000)
 
    def _transform(self, X):
        M = self.tfidf.transform(self.cv.transform(X)[:, self.keep])
        M = M.multiply(self.w[None, :]).tocsr()
        if self.svd is not None:
            M = self.svd.transform(M)
        return M
 
    def fit(self, X, y):
        from sklearn.feature_extraction.text import TfidfTransformer
 
        Xb = self.cv.fit_transform(X)
        ig, _ = information_gain(Xb, y, len(set(y)))
        k = min(self.top_k, len(ig))
        self.keep = np.argsort(ig)[-k:]
        w = ig[self.keep]
        self.w = w / w.max()
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(Xb[:, self.keep])
        M = self.tfidf.transform(Xb[:, self.keep]).multiply(self.w[None, :]).tocsr()
        if self.svd_dim:
            self.svd = TruncatedSVD(self.svd_dim, random_state=0).fit(M)
            M = self.svd.transform(M)
        self.clf.fit(M, y)
        return self
 
    def predict(self, X):
        return self.clf.predict(self._transform(X))
 
 
def _deflate_len(data: bytes, zdict: bytes | None = None) -> int:
    if zdict:
        c = zlib.compressobj(level=9, zdict=zdict)
    else:
        c = zlib.compressobj(level=9)
    return len(c.compress(data) + c.flush())
 
 
class CompressionMDL:
    """
    Classifieur MDL par compression : la classe retenue est celle dont le
    dictionnaire permet de décrire x avec le moins d'octets.
 
    mode="raw"     : dictionnaire = échantillon de textes bruts de la classe
    mode="entropy" : dictionnaire = termes les plus informatifs (poids IG) de la classe,
                     les plus importants placés en fin (distance courte = coût faible)
    """
 
    def __init__(self, mode="raw", seed=0):
        assert mode in ("raw", "entropy")
        self.mode, self.seed = mode, seed
        self.name = f"compress_mdl_{mode}"
 
    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.seed)
        self.dicts = {}
        if self.mode == "raw":
            for c in self.classes_:
                docs = [X[i] for i in np.where(y == c)[0]]
                rng.shuffle(docs)
                blob = " \n".join(docs).encode("utf8")
                self.dicts[c] = blob[-ZLIB_WINDOW:]
        else:
            cv = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True)
            Xb = cv.fit_transform(X)
            ig, p_c_t = information_gain(Xb, y.astype(int) if y.dtype.kind in "iu" else
                                         np.searchsorted(self.classes_, y), len(self.classes_))
            vocab = np.array(cv.get_feature_names_out())
            prior = np.array([(y == c).mean() for c in self.classes_])
            for j, c in enumerate(self.classes_):
                # termes sur-représentés dans la classe c, triés par IG croissant
                idx = np.where(p_c_t[:, j] > prior[j])[0]
                idx = idx[np.argsort(ig[idx])]
                blob = " ".join(vocab[idx]).encode("utf8")
                self.dicts[c] = blob[-ZLIB_WINDOW:]
        return self
 
    def predict(self, X):
        out = []
        for x in X:
            b = x.encode("utf8")
            costs = [_deflate_len(b, self.dicts[c]) for c in self.classes_]
            out.append(self.classes_[int(np.argmin(costs))])
        return np.array(out)
 
 
class TransformerBaseline:
    """Modèle HuggingFace pré-entraîné, inférence seule (pas de fine-tuning)."""
 
    def __init__(self, model_name, batch_size=32, max_len=128):
        self.model_name, self.batch_size, self.max_len = model_name, batch_size, max_len
        self.name = f"transformer:{model_name.split('/')[-1]}"
        self.n_params = None
 
    def fit(self, X, y):
        from transformers import pipeline
 
        self.pipe = pipeline("text-classification", model=self.model_name,
                             truncation=True, max_length=self.max_len)
        self.n_params = sum(p.numel() for p in self.pipe.model.parameters())
        return self
 
    def predict(self, X):
        preds = self.pipe(list(X), batch_size=self.batch_size)
        return np.array([p["label"].lower() for p in preds])
 
 
# ----------------------------------------------------------------------------
# Mesure
# ----------------------------------------------------------------------------
@dataclass
class Result:
    model: str
    accuracy: float
    f1_macro: float
    train_s: float
    infer_ms_per_1k: float
    size_bytes: int
    peak_mem_mb: float
 
 
def model_size(model) -> int:
    if isinstance(model, TransformerBaseline):
        return int(4 * (model.n_params or 0))  # float32
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1
 
 
def run(model, Xtr, ytr, Xte, yte) -> Result:
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0
 
    t0 = time.perf_counter()
    pred = model.predict(Xte)
    infer = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
 
    pred = np.asarray(pred)
    return Result(
        model=model.name,
        accuracy=accuracy_score(yte, pred),
        f1_macro=f1_score(yte, pred, average="macro"),
        train_s=train_s,
        infer_ms_per_1k=1000 * infer / len(Xte) * 1000,
        size_bytes=model_size(model),
        peak_mem_mb=peak / 1e6,
    )
 
 
def print_table(results):
    hdr = f"{'modèle':<28}{'acc':>7}{'F1':>7}{'train(s)':>10}{'ms/1k':>10}{'taille(Ko)':>12}{'mem(Mo)':>9}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in results:
        print(f"{r.model:<28}{r.accuracy:>7.3f}{r.f1_macro:>7.3f}{r.train_s:>10.2f}"
              f"{r.infer_ms_per_1k:>10.0f}{r.size_bytes / 1024:>12.0f}{r.peak_mem_mb:>9.1f}")
 
 
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="jeu synthétique (test du pipeline)")
    src.add_argument("--csv", help="fichier CSV")
    src.add_argument("--hf", help="nom d'un dataset HuggingFace")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=2000, help="nb de termes gardés (entropie)")
    ap.add_argument("--svd-dim", type=int, default=128)
    ap.add_argument("--transformer", default=None, help="ex. ProsusAI/finbert")
    ap.add_argument("--max-test", type=int, default=None, help="limite le test set (utile pour transformer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()
 
    if args.csv:
        texts, labels = load_csv(args.csv, args.text_col, args.label_col)
    elif args.hf:
        texts, labels = load_hf(args.hf, args.text_col, args.label_col, args.hf_config)
    else:
        print("[info] mode démo (données synthétiques)")
        texts, labels = load_demo(seed=args.seed)
 
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=args.test_size,
                                          random_state=args.seed, stratify=labels)
    if args.max_test:
        Xte, yte = Xte[: args.max_test], yte[: args.max_test]
 
    # Les modèles sklearn travaillent sur des labels entiers ; on garde les noms
    # pour le transformer (labels textuels) via un mapping commun.
    le = LabelEncoder().fit(labels)
    ytr_i, yte_i = le.transform(ytr), le.transform(yte)
    print(f"[info] train={len(Xtr)}  test={len(Xte)}  classes={list(le.classes_)}")
 
    models = [
        TfidfLogReg(),
        EntropyLogReg(top_k=args.top_k),
        EntropyLogReg(top_k=args.top_k, svd_dim=args.svd_dim),
        CompressionMDL("raw", args.seed),
        CompressionMDL("entropy", args.seed),
    ]
 
    results = []
    for m in models:
        print(f"[run] {m.name} ...")
        results.append(run(m, Xtr, ytr_i, Xte, yte_i))
 
    if args.transformer:
        print(f"[run] transformer {args.transformer} (inférence seule) ...")
        tm = TransformerBaseline(args.transformer)
        # labels textuels pour comparer aux sorties du modèle
        class_names = [c.lower() for c in le.classes_]
        res = run(tm, Xtr, [class_names[i] for i in ytr_i], Xte, [class_names[i] for i in yte_i])
        results.append(res)
 
    print_table(results)
    with open(args.out, "w", encoding="utf8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n[info] résultats sauvegardés dans {args.out}")
 
 
if __name__ == "__main__":
    main()
