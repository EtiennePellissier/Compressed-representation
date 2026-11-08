# Sentiment léger par entropie et compression

Ce projet compare deux approches légères, inspirées de la théorie de
l'information, à FinBERT sur les mêmes exemples de test :

1. **`entropy_weighted_nb`** : classifieur Bernoulli Naive Bayes fourni par
   [`entropy_sentiment.py`](./entropy_sentiment.py), qui sélectionne et pondère
   ses mots par leur gain d'information ;
2. **`zlib_mdl`** : classifieur par compression, qui compare le coût du modèle
   de chaque classe, amorti par exemple d'entraînement, au coût de compression
   du texte à classer et au coût de son a priori ;
3. **FinBERT** (`ProsusAI/finbert`) : modèle pré-entraîné, utilisé en inférence
   seule, sans fine-tuning sur le corpus évalué.

Le calcul du gain d'information utilisé pour la sélection des mots est :

`IG(t) = H(Y) - P(t) H(Y | t) - P(non-t) H(Y | non-t)`.

La seconde formule suit l'intuition MDL du cours — minimiser la longueur de
description du modèle et des données restantes. `zlib_mdl` est une
approximation expérimentale, pas un code universel optimal. Le classifieur
entropique, lui, n'est pas présenté comme un algorithme MDL complet.

## Lancer la comparaison

Utilisez un environnement Python 3.11–3.13 pour les dépendances PyTorch et
Transformers :

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-benchmark.txt
```

PyTorch est installé séparément pour choisir explicitement la version CPU ;
pour GPU, installez la variante PyTorch/CUDA correspondant à votre machine.
Le chargeur Financial PhraseBank lit le ZIP avec la bibliothèque standard.

Le jeu d'évaluation annoté Financial PhraseBank est disponible dans
[son dépôt Hugging Face](https://huggingface.co/datasets/takala/financial_phrasebank).
Téléchargez l'archive officielle une fois (ainsi que FinBERT au premier
lancement) :

```console
curl -L 'https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip' \
  -o /tmp/FinancialPhraseBank-v1.0.zip
python Traitement/evaluation/benchmark.py \
  --phrasebank-zip /tmp/FinancialPhraseBank-v1.0.zip \
  --agreement 75 \
  --seed 42 \
  --out results/financial_phrasebank.json
```

Le chargeur lit l'archive directement, déduplique les phrases identiques et
n'exécute aucun script distant. Financial PhraseBank est sous licence
CC BY-NC-SA 3.0 ; respectez ses conditions et citez : Malo, P. et al. (2014),
“Good debt or bad debt: Detecting semantic orientations in economic texts”,
*Journal of the Association for Information Science and Technology*, 65.

Pour vos propres annotations, fournissez un CSV UTF-8 avec les colonnes `text`
et `label` (les étiquettes pour FinBERT doivent être `positive`, `negative` ou
`neutral`). Si le CSV comporte une colonne date au format ISO-8601, demandez un
test chronologique ; les exemples d'une même journée restent ensemble :

```console
python Traitement/evaluation/benchmark.py \
  --csv data/financial_sentiment.csv \
  --text-col text \
  --label-col label \
  --split chronological \
  --time-col published_at \
  --out results/financial_sentiment.json
```

Si aucune date n'est disponible, le split stratifié aléatoire reste utilisable
avec `--seed 42` (comme dans l'exemple Financial PhraseBank ci-dessus).

Pour vérifier le pipeline sans télécharger FinBERT :

```console
python Traitement/evaluation/benchmark.py \
  --csv data/financial_sentiment.csv \
  --skip-finbert
```

Chaque exécution produit un rapport JSON détaillé (métriques par classe et
matrice de confusion) et un résumé Markdown prêt à consulter. Les rapports
indiquent l'empreinte SHA-256 du corpus, le découpage stratifié, la graine,
l'accuracy, le F1 macro, le temps d'entraînement/chargement et d'inférence, la
taille du modèle ainsi que le pic d'allocations Python suivi par `tracemalloc`.
FinBERT s'exécute sur CPU par défaut ; `--device cuda` active le premier GPU
CUDA disponible.

Les limites de chaque mesure sont précisées dans le rapport : la taille FinBERT
est celle de ses poids, sans tokenizer ni runtime, alors que la taille des
modèles légers est celle de leur état sérialisé. `tracemalloc` ne voit pas les
allocations natives PyTorch et sous-estime donc la RAM de FinBERT. Pour une
présentation, comparez le score qualité sur le même corpus et le même split,
mais présentez les tailles et allocations avec ces réserves. Les titres
AlphaVantage présents dans `Data/` ne sont pas annotés et ne permettent donc
pas de calculer une accuracy ou un F1.

Le split aléatoire stratifié facilite la comparaison, mais ne neutralise pas
les doublons ni les fuites temporelles. Pour une expérience financière finale,
dédupliquez les textes et préférez `--split chronological` avec `--time-col`.
Relancez ensuite avec plusieurs dates de coupure/corpus si vous devez évaluer
la stabilité des résultats.
FinBERT étant pré-entraîné alors que les modèles légers sont ajustés sur le
train, cette comparaison mesure des systèmes complets, pas des architectures
entraînées sur exactement la même information.

## Vérification

Le classifieur entropique et les tests du benchmark léger n'ont besoin que de la
bibliothèque standard :

```console
python -m unittest discover -s tests -v
```

Pour les détails sur le classifieur entropique, voir
[`docs/entropy-sentiment.md`](./docs/entropy-sentiment.md).

## Résultat pilote

Une exécution CPU de référence est conservée dans
[`results/financial_phrasebank.md`](./results/financial_phrasebank.md) et
[`results/financial_phrasebank.json`](./results/financial_phrasebank.json) :
Financial PhraseBank 75 % d'accord, phrases dédupliquées, split stratifié
80/20, graine 42, 2 759 exemples d'entraînement et 689 de test.

| Modèle | Accuracy | F1 macro | Inférence / 1 000 textes | Taille mesurée |
|---|---:|---:|---:|---:|
| Majorité | 0,621 | 0,255 | < 1 ms | 115 octets |
| Entropie pondérée | 0,647 | 0,321 | 6,43 s | 0,264 Mo |
| Compression `zlib_mdl` | 0,724 | 0,509 | 3,73 s | 0,099 Mo |
| FinBERT | 0,939 | 0,931 | 71,76 s | 437,94 Mo |

Ce résultat montre un compromis, pas une parité de qualité : sur ce split,
`zlib_mdl` est environ 19 fois plus rapide en inférence CPU et son état
sérialisé est beaucoup plus petit que les poids FinBERT, mais son F1 macro est
inférieur de 0,422. C'est un seul split, donc un résultat pilote ; répétez
l'expérience avec plusieurs graines avant d'en tirer une conclusion générale.
Les tailles mesurées ne représentent pas exactement le même format de
stockage, et le pic `tracemalloc` n'est pas la RAM réelle du modèle.

## Présentation orale

Le [compte-rendu oral de cinq minutes](./docs/compte-rendu-5-minutes.md) et
les sources Beamer de la [présentation Overleaf](./overleaf/main.tex) sont
fournis dans le dépôt. L'archive
[`presentation-overleaf.zip`](./presentation-overleaf.zip) s'importe
directement dans Overleaf ; les consignes sont dans
[`overleaf/README.md`](./overleaf/README.md).