"""Compact entropy-weighted sentiment classification using the Python standard library."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence


_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase Unicode words, retaining internal apostrophes."""
    if not isinstance(text, str):
        raise TypeError("each text must be a string")
    return _TOKEN_RE.findall(text.casefold())


def _entropy(counts: Sequence[int], total: int) -> float:
    if total == 0:
        return 0.0
    result = 0.0
    for count in counts:
        if count:
            probability = count / total
            result -= probability * math.log2(probability)
    return result


class EntropySentimentClassifier:
    """Bernoulli Naive Bayes with information-gain-selected, weighted features.

    Every token is represented once per document. Information gain is computed
    for the token's presence *and absence*, and the resulting bits weight the
    feature's log-likelihood in the classifier.
    """

    def __init__(self, max_features: int = 5000, alpha: float = 1.0) -> None:
        if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1:
            raise ValueError("max_features must be a positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a positive finite number")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a positive finite number")

        self.max_features = max_features
        self.alpha = float(alpha)
        self.classes_: tuple[str, ...] = ()
        self.vocabulary_: tuple[str, ...] = ()
        self.information_gain_: dict[str, float] = {}
        self._class_counts: Counter[str] = Counter()
        self._term_class_counts: dict[str, Counter[str]] = {}

    def fit(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> EntropySentimentClassifier:
        """Fit on labeled documents and return this classifier."""
        if isinstance(texts, str) or isinstance(labels, str):
            raise TypeError("texts and labels must be sequences, not strings")
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("at least one training example is required")
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("labels must be non-empty strings")

        documents = [set(tokenize(text)) for text in texts]
        class_counts: Counter[str] = Counter(labels)
        term_class_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for document, label in zip(documents, labels):
            for term in document:
                term_class_counts[term][label] += 1

        sample_count = len(labels)
        base_entropy = _entropy(list(class_counts.values()), sample_count)
        information_gain: dict[str, float] = {}
        for term, present_counts in term_class_counts.items():
            present_total = sum(present_counts.values())
            absent_total = sample_count - present_total
            absent_counts = [
                class_count - present_counts[label]
                for label, class_count in class_counts.items()
            ]
            conditional_entropy = (
                present_total / sample_count
                * _entropy(list(present_counts.values()), present_total)
                + absent_total / sample_count * _entropy(absent_counts, absent_total)
            )
            # Floating-point roundoff can make a theoretically zero gain negative.
            information_gain[term] = max(0.0, base_entropy - conditional_entropy)

        selected = sorted(
            (term for term, gain in information_gain.items() if gain > 0),
            key=lambda term: (-information_gain[term], term),
        )[: self.max_features]

        self.classes_ = tuple(sorted(class_counts))
        self.vocabulary_ = tuple(selected)
        self.information_gain_ = {term: information_gain[term] for term in selected}
        self._class_counts = class_counts
        self._term_class_counts = {term: term_class_counts[term] for term in selected}
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict one class label for each document."""
        if not self.classes_:
            raise RuntimeError("fit must be called before predict")
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence, not a string")

        total = sum(self._class_counts.values())
        vocabulary = set(self.vocabulary_)
        predictions = []
        for text in texts:
            present = vocabulary.intersection(tokenize(text))
            best_label = self.classes_[0]
            best_score = float("-inf")
            for label in self.classes_:
                class_count = self._class_counts[label]
                score = math.log(
                    (class_count + self.alpha)
                    / (total + self.alpha * len(self.classes_))
                )
                for term in self.vocabulary_:
                    probability = (
                        self._term_class_counts[term][label] + self.alpha
                    ) / (class_count + 2 * self.alpha)
                    score += self.information_gain_[term] * math.log(
                        probability if term in present else 1 - probability
                    )
                if score > best_score:
                    best_label, best_score = label, score
            predictions.append(best_label)
        return predictions
