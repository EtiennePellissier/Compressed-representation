import os
import re
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# GESTION DU CHEMIN : On force la recherche du .env à la racine du projet
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

ALPHA_VANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

stocks = [
    {"name": "Nike", "ticker": "NKE"},
    {"name": "Target", "ticker": "TGT"},
    {"name": "Disney", "ticker": "DIS"},
    {"name": "Starbucks", "ticker": "SBUX"},
    {"name": "Tesla", "ticker": "TSLA"}
]

print("Lancement de la récupération et génération des rapports HTML...\n")

output_dir = Path(__file__).resolve().parent / "AlphaVantageàtraiter"
output_dir.mkdir(parents=True, exist_ok=True)

html_output_dir = Path(__file__).resolve().parent / "RapportsHTML"
html_output_dir.mkdir(parents=True, exist_ok=True)

# Style CSS pour rendre le fichier HTML super propre et lisible
html_style = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f4f6f9; color: #333; }
    h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
    .stats { background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .card { background: white; padding: 20px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 6px solid #ccc; }
    .CONSERVE { border-left-color: #2ecc71; background-color: #f9fff9; }
    .REJETE-RELEVANCE { border-left-color: #e67e22; background-color: #fff9f5; }
    .REJETE-TEXTE { border-left-color: #e74c3c; background-color: #fff5f5; }
    .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; color: white; margin-bottom: 10px; }
    .badge-ok { background-color: #2ecc71; }
    .badge-rel { background-color: #e67e22; }
    .badge-txt { background-color: #e74c3c; }
    .meta { font-size: 0.9em; color: #7f8c8d; margin-bottom: 8px; }
    .title { font-size: 1.2em; font-weight: bold; margin: 5px 0; color: #2c3e50; }
    .summary { font-style: italic; color: #555; margin-top: 8px; background: #fdfdfd; padding: 10px; border-radius: 4px; }
</style>
"""

for index, stock in enumerate(stocks):
    name = stock["name"]
    ticker = stock["ticker"]
    
    print(f"Traitement de {name} ({ticker})...")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "sort": "LATEST",
        "limit": 200,
        "apikey": ALPHA_VANTAGE_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if "feed" not in data:
            print(f"Erreur API pour {ticker}")
            continue
            
        articles_bruts = data.get("feed", [])
        
        # Listes pour stocker temporairement TOUS les articles pour le rapport HTML
        tous_les_articles = []
        nb_conserves = 0
        nb_rejete_texte = 0
        nb_rejete_relevance = 0
        
        for item in articles_bruts:
            title = item.get("title", "")
            summary = item.get("summary", "")
            
            if not title or not summary:
                continue
            
            # 1. Vérification sécurité texte (Regex)
            text_combined = f"{title} {summary}".lower()
            has_name = name.lower() in text_combined
            has_ticker = bool(re.search(r"\b" + re.escape(ticker.lower() + r"\b"), text_combined))
            
            # Trouver le score de pertinence
            ticker_relevance = 0.0
            for ticker_info in item.get("ticker_sentiment", []):
                if ticker_info["ticker"] == ticker:
                    ticker_relevance = float(ticker_info["relevance_score"])
                    break
            
            # Détermination du statut précis pour le rapport HTML
            if not has_name and not has_ticker:
                statut = "REJETE-TEXTE"
                nb_rejete_texte += 1
            elif ticker_relevance < 0.35:
                statut = "REJETE-RELEVANCE"
                nb_rejete_relevance += 1
            else:
                statut = "CONSERVE"
                nb_conserves += 1
                
            tous_les_articles.append({
                "statut": statut,
                "score": ticker_relevance,
                "source": item.get("source"),
                "date": item.get("time_published"),
                "title": title,
                "summary": summary
            })
            
        # Tri de TOUS les articles par score décroissant pour le rendu visuel
        tous_les_articles = sorted(tous_les_articles, key=lambda x: x["score"], reverse=True)
        
        # Génération du fichier HTML de rapport dans le dossier dédié
        html_filename = html_output_dir / f"rapport_filtrage_{ticker.lower()}.html"
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(f"<html><head><meta charset='utf-8'>{html_style}<title>Rapport {ticker}</title></head><body>")
            f.write(f"<h1>Rapport d'analyse du filtre pour {name} ({ticker})</h1>")
            
            # Tableau de bord des stats en haut de page
            f.write(f"<div class='stats'>")
            f.write(f"  <p><b>Articles bruts analysés :</b> {len(articles_bruts)}</p>")
            f.write(f"  <p><span style='color:#2ecc71'>●</span> <b>Conservés (prêts pour FinBERT) :</b> {nb_conserves}</p>")
            f.write(f"  <p><span style='color:#e67e22'>●</span> <b>Rejetés - Pertinence insuffisante (< 0.35) :</b> {nb_rejete_relevance}</p>")
            f.write(f"  <p><span style='color:#e74c3c'>●</span> <b>Rejetés - Faux positifs (Nom/Ticker absent) :</b> {nb_rejete_texte}</p>")
            f.write(f"</div>")
            
            # Génération des cartes d'articles
            for art in tous_les_articles:
                if art['statut'] == "CONSERVE":
                    badge = f"<span class='badge badge-ok'>✅ CONSERVÉ — Score : {art['score']}</span>"
                elif art['statut'] == "REJETE-RELEVANCE":
                    badge = f"<span class='badge badge-rel'>⚠️ REJETÉ (PERTINENCE < 0.35) — Score : {art['score']}</span>"
                else:
                    badge = f"<span class='badge badge-txt'>❌ REJETÉ (NOM/TICKER ABSENT) — Score : {art['score']}</span>"
                    
                f.write(f"<div class='card {art['statut']}'>")
                f.write(f"  {badge}")
                f.write(f"  <div class='meta'>Source : {art['source']} | Date : {art['date']}</div>")
                f.write(f"  <div class='title'>{art['title']}</div>")
                f.write(f"  <div class='summary'>{art['summary']}</div>")
                f.write(f"</div>")
                
            f.write("</body></html>")
            
        print(f" -> Rapport HTML généré : {html_filename.name} ({nb_conserves} valides)")
        
        if index < len(stocks) - 1:
            time.sleep(15)
            
    except Exception as e:
        print(f"Erreur sur {ticker} : {e}")

print("\nTerminé ! Va dans ton dossier 'RapportsHTML' et double-clique sur les fichiers .html pour tout inspecter visuellement.")
