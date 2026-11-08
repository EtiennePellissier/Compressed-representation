# Résultats du benchmark sentiment

> Les scores dépendent du jeu de données et de la graine. Le mode démo n'est pas utilisé pour les résultats présentés.

## Protocole

- Source : `Financial PhraseBank (75% agreement)`
- Empreinte SHA-256 du corpus : `6706bb96da5c96902cc5e30e60f8542814dbf4141823ec6cad255c865536bf0b`
- Exemples : 3448 (2759 entraînement, 689 test)
- Split : stratified, 20.0% test (cible 20%), graine 42
- Python / plateforme : 3.12.3 / `Linux-6.8.0-1064-azure-x86_64-with-glibc2.39`
- Classes : negative, neutral, positive

## Comparaison

| Modèle | Accuracy | F1 macro | Entraînement / chargement (s) | Inférence (ms / 1 000 textes) | Taille enregistrée (Mo) | Pic mémoire Python (Mo) |
|---|---:|---:|---:|---:|---:|---:|
| majority_baseline | 0.6212 | 0.2554 | 0.000 | 0.00 | 0.000 | 0.006 |
| entropy_weighted_nb | 0.6473 | 0.3206 | 0.127 | 6431.89 | 0.264 | 8.897 |
| zlib_mdl | 0.7242 | 0.5085 | 0.001 | 3727.44 | 0.099 | 0.715 |
| finbert:ProsusAI/finbert | 0.9390 | 0.9308 | 0.532 | 71762.42 | 437.938 | 11.526 |

Pour FinBERT, la taille correspond aux paramètres du modèle, sans tokenizer ni runtime. Pour les modèles légers, elle correspond à l'objet entraîné sérialisé. Le pic mémoire Python est mesuré avec `tracemalloc` pendant l'entraînement ou le chargement du modèle ; il n'inclut pas les allocations natives de PyTorch ni la RAM d'inférence (c'est donc une borne basse pour FinBERT). Le temps FinBERT inclut le téléchargement éventuel et le chargement du modèle dans « Entraînement / chargement » ; FinBERT n'est pas réentraîné sur ce corpus.

Les matrices de confusion et scores par classe figurent dans le fichier JSON associé.
