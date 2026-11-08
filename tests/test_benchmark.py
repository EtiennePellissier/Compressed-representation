import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from Traitement.evaluation.benchmark import (
    CompressionMDLClassifier,
    _finbert_label,
    _validate_finbert_labels,
    classification_metrics,
    chronological_split,
    load_csv,
    load_csv_with_time,
    load_financial_phrasebank,
    stratified_split,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "Traitement" / "evaluation" / "benchmark.py"


class BenchmarkTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_preserves_every_class(self):
        labels = ["positive"] * 8 + ["negative"] * 8 + ["neutral"] * 8

        first = stratified_split(labels, test_size=0.25, seed=13)
        second = stratified_split(labels, test_size=0.25, seed=13)

        self.assertEqual(first, second)
        train, test = first
        self.assertEqual(len(test), 6)
        self.assertEqual({labels[index] for index in train}, set(labels))
        self.assertEqual({labels[index] for index in test}, set(labels))

    def test_chronological_split_keeps_days_together_and_orders_holdout(self):
        timestamps = [
            "2024-01-03T09:00:00",
            "2024-01-01",
            "2024-01-03T18:00:00",
            "2024-01-02",
            "2024-01-02T15:00:00",
            "2024-01-01T10:00:00",
        ]

        train, test = chronological_split(timestamps, test_size=0.25)

        train_dates = {timestamps[index][:10] for index in train}
        test_dates = {timestamps[index][:10] for index in test}
        self.assertEqual(train_dates, {"2024-01-01", "2024-01-02"})
        self.assertEqual(test_dates, {"2024-01-03"})
        self.assertFalse(train_dates & test_dates)

    def test_metrics_include_per_class_scores_and_confusion_matrix(self):
        result = classification_metrics(
            ["negative", "positive", "neutral"],
            ["negative", "neutral", "neutral"],
        )

        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertIn("macro_f1", result)
        self.assertEqual(result["per_class"]["positive"]["support"], 1)
        self.assertEqual(result["confusion_matrix"]["positive"]["neutral"], 1)

    def test_finbert_labels_are_mapped_and_validated(self):
        self.assertEqual(
            _finbert_label("LABEL_0", {0: "Positive"}),
            "positive",
        )
        _validate_finbert_labels(["positive", "negative", "neutral"])
        with self.assertRaisesRegex(ValueError, "unsupported labels"):
            _validate_finbert_labels(["bullish", "bearish"])

    def test_zlib_mdl_classifier_fits_and_predicts(self):
        model = CompressionMDLClassifier().fit(
            [
                "revenue growth beats expectations",
                "profits rise after strong sales",
                "losses deepen as demand falls",
                "shares drop after weak results",
            ],
            ["positive", "positive", "negative", "negative"],
        )

        predictions = model.predict(["profits rise", "losses deepen"])

        self.assertEqual(len(predictions), 2)
        self.assertEqual(set(predictions), {"positive", "negative"})
        self.assertEqual(model.classes_, ("negative", "positive"))

    def test_zlib_mdl_handles_empty_training_text(self):
        model = CompressionMDLClassifier().fit(["", ""], ["negative", "positive"])

        self.assertEqual(model.predict([""]), ["negative"])

    def test_csv_loader_checks_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.csv"
            path.write_text(
                "sentence,label\n\"good results\",positive\n", encoding="utf-8"
            )

            self.assertEqual(
                load_csv(path, "sentence", "label"),
                (["good results"], ["positive"]),
            )
            with self.assertRaisesRegex(ValueError, "must contain columns"):
                load_csv(path, "text", "label")

    def test_csv_loader_reads_iso_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dated.csv"
            path.write_text(
                "text,label,date\n"
                '"first article",positive,2024-01-01\n'
                '"second article",negative,2024-01-02\n',
                encoding="utf-8",
            )

            self.assertEqual(
                load_csv_with_time(path, "text", "label", "date"),
                (
                    ["first article", "second article"],
                    ["positive", "negative"],
                    ["2024-01-01", "2024-01-02"],
                ),
            )

    def test_phrasebank_loader_deduplicates_sentences_without_remote_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phrasebank.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
                    "profits rise@positive\n"
                    "losses deepen@negative\n"
                    "profits rise@positive\n",
                )

            self.assertEqual(
                load_financial_phrasebank(path, "75"),
                (
                    ["profits rise", "losses deepen"],
                    ["positive", "negative"],
                ),
            )

    def test_cli_writes_comparable_reports_without_finbert(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "labeled.csv"
            output_path = Path(directory) / "report.json"
            rows = ["text,label"]
            for label, phrase in (
                ("positive", "profits improve"),
                ("negative", "losses increase"),
                ("neutral", "company reports"),
            ):
                for index in range(4):
                    rows.append(f'"{phrase} {index}",{label}')
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARK),
                    "--csv",
                    str(csv_path),
                    "--skip-finbert",
                    "--out",
                    str(output_path),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [result["model"] for result in report["results"]],
                ["majority_baseline", "entropy_weighted_nb", "zlib_mdl"],
            )
            self.assertTrue(output_path.with_suffix(".md").is_file())
            self.assertEqual(report["metadata"]["test_examples"], 3)


if __name__ == "__main__":
    unittest.main()
