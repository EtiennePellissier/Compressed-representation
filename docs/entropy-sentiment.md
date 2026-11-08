# Entropy-weighted sentiment classifier

This first increment is a small, dependency-free baseline for the project's
compressed-representation goal. It uses binary word features rather than
embeddings or transformers, selects the most informative features, and weights
their Bernoulli Naive Bayes log-likelihoods by information gain.

For each token, the feature weight is

`H(label) - P(token) H(label | token) - P(not token) H(label | not token)`.

Including both presence and absence matters: a token appearing in every
document has no information gain. Feature ties are resolved by token spelling
so the selected vocabulary is reproducible. The model stores counts for the
selected vocabulary, not a dense document-feature matrix.

```python
from entropy_sentiment import EntropySentimentClassifier

model = EntropySentimentClassifier(max_features=2000)
model.fit(
    ["I love it", "I hate it", "excellent film", "awful film"],
    ["positive", "negative", "positive", "negative"],
)
print(model.predict(["love this film", "awful movie"]))
```

Evaluate a labeled CSV (`text` and `label` columns by default) with a
deterministic stratified 75/25 split. The JSON report includes accuracy,
macro-F1, per-class precision/recall/F1, a confusion matrix, and a
majority-class baseline:

```console
python evaluate_sentiment.py --csv labels.csv --text-col text --label-col label --seed 13
```

For example, the test suite uses a small, deliberately simple three-class
corpus and checks that all held-out examples are classified correctly. This is
a pipeline smoke test, not a representative benchmark: it does not demonstrate
real-world or transformer-level accuracy. Evaluate on independently labeled
data, avoid duplicate/article leakage across splits, and prefer a chronological
holdout for financial news. Compare to the majority baseline and a conventional
TF-IDF + logistic-regression baseline before claiming useful quality.

Run the standard-library tests from the repository root with:

```console
python -m unittest discover -s tests -v
```

To compare this classifier with the compression-based MDL approximation and
FinBERT on the same stratified holdout, follow the dataset and benchmark
instructions in the [project README](../README.md). FinBERT and the Hugging
Face model dependencies are optional; the entropy classifier itself does not
require them.

The tokenizer marks a three-token scope after simple negation cues, which helps
with phrases such as “not very good”; it is not a full linguistic negation
parser. The estimator uses information gain for feature selection and
weighting; it does not yet optimize a complete minimum-description-length
objective over model and residual-data costs. Measure accuracy, model size,
and runtime on the same held-out dataset before making performance claims.
