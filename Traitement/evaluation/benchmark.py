#!/usr/bin/env python3
"""Compare the entropy classifier and a zlib MDL baseline with FinBERT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import random
import sys
import time
import tracemalloc
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entropy_sentiment import EntropySentimentClassifier  # noqa: E402


_FINBERT_LABELS = {"positive", "negative", "neutral"}
_ZLIB_DICTIONARY_SIZE = 32_768


def stratified_split(
    labels: Sequence[str], test_size: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Make one deterministic holdout while keeping every class in both sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not labels:
        raise ValueError("the dataset must contain labeled examples")

    by_class: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if not isinstance(label, str) or not label.strip():
            raise ValueError("labels must be non-empty strings")
        by_class.setdefault(label.strip(), []).append(index)

    rng = random.Random(seed)
    train: list[int] = []
    test: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < 2:
            raise ValueError(
                f"class {label!r} needs at least 2 examples for stratified evaluation"
            )
        rng.shuffle(indices)
        test_count = min(
            len(indices) - 1, max(1, round(len(indices) * test_size))
        )
        test.extend(indices[:test_count])
        train.extend(indices[test_count:])
    return sorted(train), sorted(test)


def chronological_split(
    timestamps: Sequence[str], test_size: float = 0.2
) -> tuple[list[int], list[int]]:
    """Reserve the latest dates for testing without splitting a calendar day."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(timestamps) < 2:
        raise ValueError("at least two dated examples are required")

    indices_by_date: dict[str, list[int]] = {}
    for index, value in enumerate(timestamps):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise ValueError(
                f"timestamps must be ISO-8601 dates or datetimes; got {value!r}"
            ) from error
        indices_by_date.setdefault(parsed.date().isoformat(), []).append(index)

    ordered_dates = sorted(indices_by_date)
    if len(ordered_dates) < 2:
        raise ValueError("chronological split requires examples from at least 2 dates")

    target_test_count = max(1, round(len(timestamps) * test_size))
    test_dates = set()
    test_count = 0
    for date in reversed(ordered_dates[1:]):
        test_dates.add(date)
        test_count += len(indices_by_date[date])
        if test_count >= target_test_count:
            break

    train = [
        index
        for date in ordered_dates
        if date not in test_dates
        for index in indices_by_date[date]
    ]
    test = [
        index
        for date in ordered_dates
        if date in test_dates
        for index in indices_by_date[date]
    ]
    return sorted(train), sorted(test)


def load_csv(path: Path, text_column: str, label_column: str) -> tuple[list[str], list[str]]:
    """Load labeled UTF-8 CSV data without requiring pandas."""
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
            if row.get(text_column, "").strip() and row.get(label_column, "").strip()
        ]
    if not rows:
        raise ValueError("CSV contains no rows with both text and label")
    return (
        [row[text_column].strip() for row in rows],
        [row[label_column].strip() for row in rows],
    )


def load_csv_with_time(
    path: Path, text_column: str, label_column: str, time_column: str
) -> tuple[list[str], list[str], list[str]]:
    """Load non-empty text, label, and ISO-8601 timestamp columns from CSV."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = (text_column, label_column, time_column)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in required
        ):
            raise ValueError(f"CSV must contain columns {required!r}")
        rows = [
            row
            for row in reader
            if all(row.get(column, "").strip() for column in required)
        ]
    if not rows:
        raise ValueError("CSV contains no rows with text, label, and timestamp")
    return (
        [row[text_column].strip() for row in rows],
        [row[label_column].strip() for row in rows],
        [row[time_column].strip() for row in rows],
    )


def load_financial_phrasebank(
    path: Path, agreement: str = "75"
) -> tuple[list[str], list[str]]:
    """Read the original Financial PhraseBank archive without remote code."""
    agreement_name = "All" if agreement == "all" else agreement
    expected_name = f"Sentences_{agreement_name}Agree.txt"
    try:
        with ZipFile(path) as archive:
            matches = [
                name
                for name in archive.namelist()
                if Path(name).name.casefold() == expected_name.casefold()
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"archive must contain exactly one {expected_name!r} file"
                )
            lines = archive.read(matches[0]).decode("iso-8859-1").splitlines()
    except BadZipFile as error:
        raise ValueError(f"{path} is not a valid Financial PhraseBank ZIP archive") from error

    texts = []
    labels = []
    seen: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            text, label = line.rsplit("@", 1)
        except ValueError as error:
            raise ValueError(
                f"invalid Financial PhraseBank row at line {line_number}"
            ) from error
        text, label = text.strip(), label.strip().casefold()
        if not text or label not in _FINBERT_LABELS:
            raise ValueError(
                f"invalid Financial PhraseBank text or label at line {line_number}"
            )
        if text in seen:
            if seen[text] != label:
                raise ValueError(
                    f"conflicting labels for duplicate sentence at line {line_number}"
                )
            continue
        seen[text] = label
        texts.append(text)
        labels.append(label)
    if not texts:
        raise ValueError(f"{expected_name} contains no labeled sentences")
    return texts, labels


def classification_metrics(actual: Sequence[str], predicted: Sequence[str]) -> dict:
    """Return accuracy, macro-F1, per-class scores, and a confusion matrix."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("metrics require at least one test example")

    classes = sorted(set(actual) | set(predicted))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores = []
    for label in classes:
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
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
        f1_scores.append(f1)

    confusion = {
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
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


class CompressionMDLClassifier:
    """Classify by amortized model description length plus zlib residual cost.

    Each class dictionary is built exclusively from its training documents.
    The score is the compressed dictionary cost per training example, plus the
    compressed test-document cost given that dictionary and the class-prior
    code length. This is a practical MDL approximation, not an optimal code.
    """

    def __init__(self) -> None:
        self.classes_: tuple[str, ...] = ()
        self._dictionaries: dict[str, bytes] = {}
        self._class_counts: Counter[str] = Counter()
        self.name = "zlib_mdl"

    def fit(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> CompressionMDLClassifier:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("at least one training example is required")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must be strings")
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("labels must be non-empty strings")

        class_counts: Counter[str] = Counter(labels)
        dictionaries = {}
        for label in sorted(class_counts):
            class_texts = [
                text for text, example_label in zip(texts, labels)
                if example_label == label
            ]
            dictionary = "\n".join(class_texts).encode("utf-8")
            dictionaries[label] = dictionary[-_ZLIB_DICTIONARY_SIZE:]

        self.classes_ = tuple(sorted(class_counts))
        self._class_counts = class_counts
        self._dictionaries = dictionaries
        return self

    @staticmethod
    def _conditional_size(text: bytes, dictionary: bytes) -> int:
        compressor = (
            zlib.compressobj(level=9, zdict=dictionary)
            if dictionary
            else zlib.compressobj(level=9)
        )
        return len(compressor.compress(text) + compressor.flush())

    def predict(self, texts: Sequence[str]) -> list[str]:
        if not self.classes_:
            raise RuntimeError("fit must be called before predict")
        total = sum(self._class_counts.values())
        predictions = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("each text must be a string")
            encoded = text.encode("utf-8")
            scored_classes = []
            for label in self.classes_:
                dictionary = self._dictionaries[label]
                model_cost = len(zlib.compress(dictionary, level=9)) / self._class_counts[label]
                residual_cost = self._conditional_size(encoded, dictionary)
                prior_cost = -math.log2(self._class_counts[label] / total) / 8
                scored_classes.append(
                    (model_cost + residual_cost + prior_cost, label)
                )
            predictions.append(min(scored_classes)[1])
        return predictions


def _measure_entropy(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    max_features: int,
) -> dict:
    model = EntropySentimentClassifier(max_features=max_features)
    start = time.perf_counter()
    model.fit(train_texts, train_labels)
    fit_seconds = time.perf_counter() - start

    memory_model = EntropySentimentClassifier(max_features=max_features)
    tracemalloc.start()
    try:
        memory_model.fit(train_texts, train_labels)
        _, peak_python_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    start = time.perf_counter()
    predictions = model.predict(test_texts)
    inference_seconds = time.perf_counter() - start
    return {
        "model": "entropy_weighted_nb",
        "fit_seconds": fit_seconds,
        "inference_ms_per_1000": inference_seconds * 1_000_000 / len(test_texts),
        "stored_model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
        "peak_python_memory_bytes": peak_python_bytes,
        "selected_features": len(model.vocabulary_),
        "metrics": classification_metrics(test_labels, predictions),
    }


def _measure_mdl(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
) -> dict:
    model = CompressionMDLClassifier()
    start = time.perf_counter()
    model.fit(train_texts, train_labels)
    fit_seconds = time.perf_counter() - start

    memory_model = CompressionMDLClassifier()
    tracemalloc.start()
    try:
        memory_model.fit(train_texts, train_labels)
        _, peak_python_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    start = time.perf_counter()
    predictions = model.predict(test_texts)
    inference_seconds = time.perf_counter() - start
    return {
        "model": model.name,
        "fit_seconds": fit_seconds,
        "inference_ms_per_1000": inference_seconds * 1_000_000 / len(test_texts),
        "stored_model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
        "peak_python_memory_bytes": peak_python_bytes,
        "selected_features": None,
        "metrics": classification_metrics(test_labels, predictions),
    }


def _measure_majority(
    train_labels: Sequence[str], test_labels: Sequence[str]
) -> dict:
    start = time.perf_counter()
    counts = Counter(train_labels)
    majority_label = min(counts, key=lambda label: (-counts[label], label))
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = [majority_label] * len(test_labels)
    inference_seconds = time.perf_counter() - start

    tracemalloc.start()
    try:
        memory_counts = Counter(train_labels)
        memory_label = min(
            memory_counts, key=lambda label: (-memory_counts[label], label)
        )
        _ = [memory_label] * len(test_labels)
        _, peak_python_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    model_state = {"counts": counts, "prediction": majority_label}
    return {
        "model": "majority_baseline",
        "fit_seconds": fit_seconds,
        "inference_ms_per_1000": inference_seconds * 1_000_000 / len(test_labels),
        "stored_model_bytes": len(
            pickle.dumps(model_state, protocol=pickle.HIGHEST_PROTOCOL)
        ),
        "peak_python_memory_bytes": peak_python_bytes,
        "selected_features": None,
        "metrics": classification_metrics(test_labels, predictions),
    }


def _finbert_label(label: str, id2label: dict) -> str:
    normalized = label.strip().casefold()
    if normalized in _FINBERT_LABELS:
        return normalized
    if normalized.startswith("label_"):
        try:
            normalized = str(id2label[int(normalized.removeprefix("label_"))]).casefold()
        except (KeyError, ValueError):
            pass
    normalized = normalized.replace("_", " ").strip()
    if normalized in _FINBERT_LABELS:
        return normalized
    raise ValueError(
        f"FinBERT returned the unrecognized class {label!r}; "
        "use a checkpoint with positive/negative/neutral labels."
    )


def _validate_finbert_labels(labels: Sequence[str]) -> None:
    unsupported_labels = sorted(
        {label.casefold() for label in labels} - _FINBERT_LABELS
    )
    if unsupported_labels:
        raise ValueError(
            "FinBERT comparison requires sentiment labels named positive, "
            f"negative, or neutral; unsupported labels: {unsupported_labels}"
        )


def _measure_finbert(
    model_name: str,
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> dict:
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "FinBERT requires the optional packages in requirements-benchmark.txt"
        ) from error

    target_device = -1
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        target_device = 0

    tracemalloc.start()
    try:
        start = time.perf_counter()
        model_path = (
            model_name
            if Path(model_name).is_dir()
            else snapshot_download(repo_id=model_name)
        )
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            truncation=True,
            max_length=max_length,
            device=target_device,
        )
        setup_seconds = time.perf_counter() - start
        _, peak_python_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    start = time.perf_counter()
    raw_predictions = classifier(list(test_texts), batch_size=batch_size)
    inference_seconds = time.perf_counter() - start

    id2label = classifier.model.config.id2label
    predictions = [
        _finbert_label(result["label"], id2label) for result in raw_predictions
    ]
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in classifier.model.parameters()
    )
    return {
        "model": f"finbert:{model_name}",
        "fit_seconds": 0.0,
        "setup_seconds": setup_seconds,
        "inference_ms_per_1000": inference_seconds * 1_000_000 / len(test_texts),
        "stored_model_bytes": parameter_bytes,
        "stored_model_size_note": "model parameter bytes; excludes tokenizer and runtime",
        "model_revision": getattr(classifier.model.config, "_commit_hash", None),
        "peak_python_memory_bytes": peak_python_bytes,
        "selected_features": None,
        "metrics": classification_metrics(test_labels, predictions),
        "device": device,
    }


def _dataset_digest(texts: Sequence[str], labels: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text, label in zip(texts, labels):
        for value in (text.encode("utf-8"), label.encode("utf-8")):
            digest.update(len(value).to_bytes(8, byteorder="big"))
            digest.update(value)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _markdown_report(metadata: dict, results: Sequence[dict]) -> str:
    rows = [
        "# Résultats du benchmark sentiment",
        "",
        "> Les scores dépendent du jeu de données et de la graine. "
        "Le mode démo n'est pas utilisé pour les résultats présentés.",
        "",
        "## Protocole",
        "",
        f"- Source : `{metadata['dataset']}`",
        f"- Empreinte SHA-256 du corpus : `{metadata['dataset_sha256']}`",
        f"- Exemples : {metadata['examples']} "
        f"({metadata['train_examples']} entraînement, {metadata['test_examples']} test)",
        f"- Split : {metadata['split_strategy']}, "
        f"{metadata['test_examples'] / metadata['examples']:.1%} test "
        f"(cible {metadata['test_size']:.0%}), graine {metadata['seed']}",
        f"- Python / plateforme : {metadata['python']} / `{metadata['platform']}`",
        f"- Classes : {', '.join(metadata['classes'])}",
        "",
        "## Comparaison",
        "",
        "| Modèle | Accuracy | F1 macro | Entraînement / chargement (s) | Inférence (ms / 1 000 textes) | Taille enregistrée (Mo) | Pic mémoire Python (Mo) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        size_mb = result["stored_model_bytes"] / 1_000_000
        memory_mb = result["peak_python_memory_bytes"] / 1_000_000
        fit_seconds = result.get("setup_seconds", result["fit_seconds"])
        metrics = result["metrics"]
        rows.append(
            f"| {result['model']} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {fit_seconds:.3f} | "
            f"{result['inference_ms_per_1000']:.2f} | {size_mb:.3f} | "
            f"{memory_mb:.3f} |"
        )
    rows.extend(
        [
            "",
            "Pour FinBERT, la taille correspond aux paramètres du modèle, sans "
            "tokenizer ni runtime. Pour les modèles légers, elle correspond à "
            "l'objet entraîné sérialisé. Le pic mémoire Python est mesuré avec "
            "`tracemalloc` pendant l'entraînement ou le chargement du modèle ; "
            "il n'inclut pas les allocations natives de PyTorch ni la RAM "
            "d'inférence (c'est donc une borne basse pour FinBERT). Le temps FinBERT inclut "
            "le téléchargement éventuel et le chargement du modèle dans "
            "« Entraînement / chargement » ; FinBERT "
            "n'est pas réentraîné sur ce corpus.",
            "",
            "Les matrices de confusion et scores par classe figurent dans le "
            "fichier JSON associé.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV contenant textes et étiquettes")
    source.add_argument(
        "--phrasebank-zip",
        type=Path,
        help="archive Financial PhraseBank téléchargée depuis Hugging Face",
    )
    parser.add_argument(
        "--agreement",
        choices=("all", "75", "66", "50"),
        default="75",
        help="seuil d'accord des annotateurs pour Financial PhraseBank",
    )
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--split",
        choices=("stratified", "chronological"),
        default="stratified",
        help="chronological requires a CSV and --time-col",
    )
    parser.add_argument("--time-col", default=None, help="ISO date/datetime column")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--finbert", default="ProsusAI/finbert")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-finbert", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/benchmark.json"),
        help="rapport JSON (un rapport Markdown voisin est aussi créé)",
    )
    args = parser.parse_args()

    if args.max_features < 1:
        parser.error("--max-features must be positive")
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive")

    timestamps = None
    if args.split == "chronological" and (not args.csv or not args.time_col):
        parser.error("--split chronological requires --csv and --time-col")
    if args.time_col and args.split != "chronological":
        parser.error("--time-col can only be used with --split chronological")

    if args.phrasebank_zip:
        if args.time_col or args.split == "chronological":
            parser.error("chronological splits require a CSV and --time-col")
        texts, labels = load_financial_phrasebank(args.phrasebank_zip, args.agreement)
        dataset_source = f"Financial PhraseBank ({args.agreement}% agreement)"
    elif args.csv and args.time_col:
        texts, labels, timestamps = load_csv_with_time(
            args.csv, args.text_col, args.label_col, args.time_col
        )
        dataset_source = str(args.csv)
    elif args.csv:
        texts, labels = load_csv(args.csv, args.text_col, args.label_col)
        dataset_source = str(args.csv)

    if any(not text.strip() for text in texts):
        raise ValueError("texts must not be empty")
    normalized_labels = [label.strip().casefold() for label in labels]
    if not args.skip_finbert:
        _validate_finbert_labels(normalized_labels)
    if args.split == "chronological":
        train_indices, test_indices = chronological_split(
            timestamps, test_size=args.test_size
        )
    else:
        train_indices, test_indices = stratified_split(
            normalized_labels, test_size=args.test_size, seed=args.seed
        )
    train_texts = [texts[index] for index in train_indices]
    train_labels = [normalized_labels[index] for index in train_indices]
    test_texts = [texts[index] for index in test_indices]
    test_labels = [normalized_labels[index] for index in test_indices]

    results = [
        _measure_majority(train_labels, test_labels),
        _measure_entropy(
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            args.max_features,
        ),
        _measure_mdl(train_texts, train_labels, test_texts, test_labels),
    ]
    if not args.skip_finbert:
        results.append(
            _measure_finbert(
                args.finbert,
                test_texts,
                test_labels,
                args.batch_size,
                args.max_length,
                args.device,
            )
        )

    metadata = {
        "dataset": dataset_source,
        "dataset_sha256": _dataset_digest(texts, normalized_labels),
        "examples": len(texts),
        "train_examples": len(train_indices),
        "test_examples": len(test_indices),
        "test_size": args.test_size,
        "split_strategy": args.split,
        "time_column": args.time_col,
        "seed": args.seed,
        "classes": sorted(set(normalized_labels)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "configuration": {
            "max_features": args.max_features,
            "finbert": None if args.skip_finbert else args.finbert,
            "finbert_batch_size": args.batch_size,
            "finbert_max_length": args.max_length,
            "device": None if args.skip_finbert else args.device,
        },
        "package_versions": {
            package: _package_version(package) for package in ("torch", "transformers")
        },
    }
    report = {"metadata": metadata, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(_markdown_report(metadata, results), encoding="utf-8")

    print(_markdown_report(metadata, results))
    print(f"JSON : {args.out}")
    print(f"Markdown : {markdown_path}")


if __name__ == "__main__":
    main()
