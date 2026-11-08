"""Reproducible CSV evaluation for the entropy sentiment baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

from entropy_sentiment import EntropySentimentClassifier


def stratified_split(
    labels: Sequence[str], test_size: float = 0.25, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Create a deterministic stratified split with each class in both sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not labels:
        raise ValueError("at least one labeled example is required")

    by_class: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if not isinstance(label, str) or not label:
            raise ValueError("labels must be non-empty strings")
        by_class.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < 2:
            raise ValueError(
                f"class {label!r} needs at least 2 examples for stratified evaluation"
            )
        rng.shuffle(indices)
        test_count = min(len(indices) - 1, max(1, round(len(indices) * test_size)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    return sorted(train_indices), sorted(test_indices)


def classification_metrics(
    actual: Sequence[str], predicted: Sequence[str], classes: Sequence[str]
) -> dict:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("metrics require at least one test example")
    if not classes:
        raise ValueError("metrics require at least one class")

    per_class = {}
    f1_values = []
    for label in classes:
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(a == label for a in actual),
        }
        f1_values.append(f1)

    matrix = {
        actual_label: {
            predicted_label: sum(
                a == actual_label and p == predicted_label
                for a, p in zip(actual, predicted)
            )
            for predicted_label in classes
        }
        for actual_label in classes
    }
    return {
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def evaluate(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    test_size: float = 0.25,
    seed: int = 0,
    max_features: int = 5000,
) -> dict:
    """Compare the classifier with a training-set majority-class baseline."""
    if isinstance(texts, str) or isinstance(labels, str):
        raise TypeError("texts and labels must be sequences, not strings")
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("texts must be strings")
    train, test = stratified_split(labels, test_size=test_size, seed=seed)
    train_texts = [texts[index] for index in train]
    train_labels = [labels[index] for index in train]
    test_texts = [texts[index] for index in test]
    test_labels = [labels[index] for index in test]

    model = EntropySentimentClassifier(max_features=max_features).fit(
        train_texts, train_labels
    )
    predictions = model.predict(test_texts)
    train_counts = Counter(train_labels)
    baseline_label = min(train_counts, key=lambda label: (-train_counts[label], label))
    classes = sorted(set(labels))
    return {
        "examples": len(labels),
        "train_examples": len(train),
        "test_examples": len(test),
        "seed": seed,
        "selected_features": len(model.vocabulary_),
        "classifier": classification_metrics(test_labels, predictions, classes),
        "majority_baseline": classification_metrics(
            test_labels, [baseline_label] * len(test_labels), classes
        ),
    }


def load_csv(
    path: Path, text_column: str, label_column: str
) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if (
            reader.fieldnames is None
            or text_column not in reader.fieldnames
            or label_column not in reader.fieldnames
        ):
            raise ValueError(
                f"CSV must contain columns {text_column!r} and {label_column!r}"
            )
        rows = [
            row
            for row in reader
            if row.get(text_column) and row.get(label_column)
        ]
    if not rows:
        raise ValueError("CSV contains no rows with both text and label")
    return [row[text_column] for row in rows], [row[label_column] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()
    texts, labels = load_csv(args.csv, args.text_col, args.label_col)
    result = evaluate(
        texts,
        labels,
        test_size=args.test_size,
        seed=args.seed,
        max_features=args.max_features,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
