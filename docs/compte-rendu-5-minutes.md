# Compte-rendu oral — environ 5 minutes

## Diapositive 1 — Problématique (30 secondes)

Bonjour. Ce projet pose une question simple : peut-on classer le sentiment
d’un texte financier sans dépendre entièrement d’un modèle lourd comme
FinBERT ? En nous inspirant du cours sur l’information algorithmique et le
principe MDL, nous avons comparé des modèles légers à FinBERT selon trois
critères : qualité, vitesse et taille.

## Diapositive 2 — Ce qui a été implémenté (45 secondes)

La première méthode représente chaque texte par des mots présents ou absents.
Elle sélectionne les mots par gain d’information puis les pondère dans un
classifieur Bernoulli Naive Bayes. Elle évite donc les embeddings denses.

La deuxième méthode applique une approximation pratique du MDL : elle
construit, à partir des textes d’entraînement, un dictionnaire de compression
zlib pour chaque sentiment. Pour classer un texte, elle compare le coût du
dictionnaire, amorti par sa classe, au coût de compression du texte et à
l’a priori de cette classe. Ce n’est pas un code optimal.

Nous avons intégré ces méthodes à un benchmark commun avec une baseline
majorité et FinBERT préentraîné, sans fine-tuning, ainsi que des rapports
reproductibles et des tests.

## Diapositive 3 — Protocole (40 secondes)

Nous avons utilisé Financial PhraseBank, avec un accord d’annotation d’au
moins 75 %. Après déduplication, il reste 3 448 phrases en trois classes.
Le split stratifié, graine 42, comprend 2 759 textes d’entraînement et 689 de
test. Tous les modèles sont comparés sur ces mêmes textes ; FinBERT tourne sur
CPU avec une limite de 128 tokens.

Nous présentons l’accuracy et le F1 macro. Ce dernier équilibre les classes,
ce qui importe ici : le test compte 428 exemples neutres, 177 positifs et
84 négatifs. L’accuracy seule pourrait masquer une mauvaise détection des
classes minoritaires.

## Diapositive 4 — Qualité obtenue (55 secondes)

La baseline prédit toujours « neutre » : son accuracy atteint déjà 62,1 %,
mais son F1 macro n’est que de 25,5 %. Elle ne trouve aucun exemple négatif ou
positif.

Le modèle entropique obtient 64,7 % d’accuracy et 32,1 % de F1 macro. Il ne
détecte toujours aucun négatif. Le zlib MDL progresse à 72,4 % d’accuracy et
50,9 % de F1 macro, mais son rappel des négatifs reste limité à 9,5 %.

FinBERT atteint 93,9 % d’accuracy et 93,1 % de F1 macro, avec un rappel élevé
dans chacune des classes. Sur cette expérience, il est donc le meilleur
modèle en qualité. L’accuracy ne raconte toutefois qu’une partie de l’histoire :
les scores par classe et les matrices de confusion sont aussi fournis dans le
rapport.

## Diapositive 5 — Coût et compromis (45 secondes)

Sur CPU, le rapport mesure environ 71,8 secondes d’inférence par millier de
textes pour FinBERT, contre 3,7 secondes pour zlib MDL, soit environ 19 fois
moins pour ce dernier sur cette machine. La taille enregistrée est de 0,099
Mo pour zlib contre 437,94 Mo pour les poids FinBERT.

Ces tailles ne couvrent pas exactement les mêmes éléments : l’objet léger
sérialisé d’un côté, les poids seuls de FinBERT de l’autre. En plus,
`tracemalloc` ne voit pas la mémoire native de PyTorch. Ces mesures suggèrent
un avantage de coût pour la compression, mais ne constituent pas une mesure
complète de la RAM en production.

## Diapositive 6 — Livrables et limites (40 secondes)

Le projet fournit le classifieur entropique, le classifieur zlib MDL, le
benchmark FinBERT, des rapports JSON et Markdown, un chargeur de corpus qui
évite d’exécuter du code distant et un split chronologique pour les données
datées. Vingt-deux tests passent sous les deux environnements Python vérifiés.

Cela reste un résultat pilote : un seul split aléatoire sur un corpus de
phrases courtes. FinBERT est préentraîné, tandis que les méthodes légères
apprennent sur le train ; ils n’ont donc pas reçu exactement la même
information. Nous ne concluons pas à une équivalence ni à une généralisation
hors de ce jeu de données.

## Diapositive 7 — Conclusion (40 secondes)

En conclusion, ces résultats mettent en évidence un compromis : zlib MDL est
plus compact et rapide, mais son F1 macro reste inférieur de 0,42 à celui de
FinBERT, notamment en raison des exemples négatifs manqués. Pour le meilleur
score, FinBERT gagne ; pour les ressources, la compression mérite d’être
améliorée.

La suite est de répéter l’expérience avec plusieurs graines, d’améliorer la
classe négative et d’évaluer sur des actualités dédupliquées avec un test
chronologique. Il faudra alors remesurer la RAM réelle avant de conclure à un
avantage opérationnel. Merci.

---

Les résultats chiffrés et les métadonnées de l’expérience sont dans
[`../results/financial_phrasebank.md`](../results/financial_phrasebank.md) et
[`../results/financial_phrasebank.json`](../results/financial_phrasebank.json).
