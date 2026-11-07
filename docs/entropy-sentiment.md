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

Run the standard-library tests from the repository root with:

```console
python -m unittest discover -s tests -v
```

This is an initial baseline, not a claim of parity with transformer accuracy.
Measure accuracy, model size, and runtime on the same held-out dataset before
making performance claims. The current estimator uses information gain for
feature selection and weighting; it does not yet optimize a complete
minimum-description-length objective over model and residual-data costs.
