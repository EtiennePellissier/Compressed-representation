import torch
import torch.nn.functional as F     # Fonctions utilitaires (comme softmax)
import pandas as pd     # Manipulation de données (Dataframe)
from tqdm import tqdm    # Barre de progression pour les boucles longues
from transformers import BertTokenizer, BertForSequenceClassification
import json

# On va chercher le fichier config
import sys
import os
# Remonte de 3 niveaux pour atteindre le dossier principal (Portfolio-Management-MODS)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)
from config import FINBERT_MODEL, MAX_TOKENS    # variables de configuration définies dans un fichier annexe config.py (dans le dossier principal) --> MAX_TOKENS = longueur maximal des textes

# ========================
# Chargement lazy du modèle (une seule fois, pour éviter de le faire à chaque appel de fonction)
# ========================

# Variables qui stockent le modèle (si elles sont égales à "None" c'est que le modèle n'a pas encore été chargé) :
_tokenizer = None
_model     = None


def _load_model():
    """Charge FinBERT en mémoire (une seule fois, lors du premier appel)."""
    global _tokenizer, _model
    if _model is None:
        print(f"[FinBERT] Chargement du modèle {FINBERT_MODEL}...")
        _tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)  # objet qui découpe le texte en tokens et le formate pour le modèle
        _model     = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)  # modèle Finbert pré-entraîné qui prend des tokens en entrées et prédit les probas de sentiment
        _model.eval()  # désactive le mode entraînement et active le mode évaluation
        print("[FinBERT] Modèle prêt.")


# ========================
# Analyse d'un seul texte
# ========================

def analyze_sentiment(text: str) -> dict:
    """
    Analyse le sentiment d'un texte financier avec FinBERT.

    Args:
        text: texte brut (titre + corps d'article)

    Returns:
        dict avec les probabilités : {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
    """
    _load_model() # charge le modèle la première fois qu'elle est appelée

    inputs = _tokenizer(
        text,
        return_tensors="pt",   # retourne des tenseurs PyTorch compatibles avec le modèle
        truncation=True,       # Coupe le texte si > MAX_TOKENS
        max_length=MAX_TOKENS,
        padding=True,      # Ajoute des zéros pour uniformiser la taille
    )

    with torch.no_grad():
        outputs = _model(**inputs)  # on passe les tokens créés dans le modèle

    probs  = F.softmax(outputs.logits, dim=-1).squeeze()   # softmax permet de transformer la sortie du modèle (logits=scores bruts) en probabilités; squeeze nettoie les dimensions inutiles de tableaux/listes
    labels = ["positive", "negative", "neutral"]

    return {label: round(probs[i].item(), 4) for i, label in enumerate(labels)}


# ========================
# Analyse par batch (plusieurs textes en même temps, donc plus rapide)
# ========================

def analyze_sentiment_batch(texts: list, batch_size: int = 8) -> list:      # Même rôle et fonctionnement que la fonction "analyze_sentiment", mais en plus rapide
    """
    Analyse une liste de textes par batch pour accélérer le traitement.

    Returns:
        Liste de dicts de sentiment, dans le même ordre que texts
    """
    _load_model()
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        )
        with torch.no_grad():
            outputs = _model(**inputs)

        probs_batch = F.softmax(outputs.logits, dim=-1)
        labels      = ["positive", "negative", "neutral"]

        for probs in probs_batch:
            results.append({label: round(probs[i].item(), 4) for i, label in enumerate(labels)})

    return results


# ========================
# Score agrégé par ticker (lettres correspondant à l'action boursière d'une entreprise)
# ========================

def compute_ticker_sentiment(articles: dict) -> pd.DataFrame:
    """
    Calcule un score de sentiment agrégé pour chaque ticker.

    Args:
        articles: dict { "AAPL": ["texte1", "texte2", ...], "MSFT": [...] }
                  (format de sortie de fetch_news.py)

    Returns:
        DataFrame (tableau) avec colonnes : ticker | sentiment_score | n_articles
        sentiment_score ∈ [-1, +1] :  -1 = très négatif, +1 = très positif, c'est la moyenne sur le nombre d'articles
    """
    results = []

    for ticker, texts in tqdm(articles.items(), desc="Analyse sentiment"): #tqdm affiche la barre de progression avec le texte "Analyse sentiment" à côté de la barre
        if not texts:
            print(f"[FinBERT] Aucun article pour {ticker}, score = 0.0")
            results.append({"ticker": ticker, "sentiment_score": 0.0, "n_articles": 0})
            continue

        sentiments  = analyze_sentiment_batch(texts)        #on utilise la fonction avec batch
        net_scores  = [s["positive"] - s["negative"] for s in sentiments] # calcule le score net de chaque article en soustrayant sa probabilité négative de sa probabilité positive
        mean_score  = sum(net_scores) / len(net_scores)     # on moyenne tous les scores nets de chaque article par ticker

        results.append({
            "ticker":          ticker,
            "sentiment_score": round(mean_score, 4),
            "n_articles":      len(texts),
        })

    df = pd.DataFrame(results).sort_values("ticker").reset_index(drop=True)   # on créé un dataframe avec les résultats, triés par ordre alphabétique de ticker
    return df


# ========================
# Test rapide
# ========================

if __name__ == "__main__":
    sample_articles = {
        "AAPL": [
            "Apple reports record quarterly earnings, beating analyst expectations by a wide margin.",
            "iPhone sales disappoint analysts this quarter amid slowing consumer demand.",
        ],
        "MSFT": [
            "Microsoft Azure cloud revenue surges 30%, driven by AI adoption across enterprises."
        ],
        "JPM": [
            "JPMorgan raises dividend as profits hit record high on strong trading revenues.",
            "JPMorgan faces regulatory scrutiny over compliance failures in overseas operations.",
        ],
    }

    #chargement des textes à partir d'un fichier json externe
    json_path = os.path.join(root_path, "Data", "AlphaVantage", "alphavantage_merged_20260610.json")
    print(f"Chargement des données depuis : {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        articles_data = json.load(f)

    df = compute_ticker_sentiment(articles_data)
    output_dir = os.path.join(root_path, "data_output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "sentiment_scores.csv")
    df.to_csv(output_path, index=False)

    print(f"Scores sauvegardés dans : {output_path}")
    print("\n=== Scores de sentiment ===")
    print(df.to_string(index=False))
