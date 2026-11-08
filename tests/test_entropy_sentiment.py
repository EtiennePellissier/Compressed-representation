import unittest
import tempfile
from pathlib import Path

from entropy_sentiment import EntropySentimentClassifier, tokenize
from evaluate_sentiment import (
    classification_metrics,
    evaluate,
    load_csv,
    stratified_split,
)


class EntropySentimentClassifierTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "I love this film",
            "I love an excellent movie",
            "I hate this film",
            "I hate a terrible movie",
        ]
        self.labels = ["positive", "positive", "negative", "negative"]

    def test_information_gain_accounts_for_presence_and_absence(self):
        model = EntropySentimentClassifier().fit(self.texts, self.labels)

        self.assertAlmostEqual(model.information_gain_["love"], 1.0)
        self.assertAlmostEqual(model.information_gain_["hate"], 1.0)
        self.assertNotIn("film", model.vocabulary_)

    def test_predicts_sentiment_and_uses_class_prior_without_known_features(self):
        model = EntropySentimentClassifier().fit(self.texts, self.labels)

        self.assertEqual(
            model.predict(["LOVE this", "hate this", "brand new tokens"]),
            ["positive", "negative", "negative"],
        )

    def test_limits_features_with_deterministic_tie_breaking(self):
        model = EntropySentimentClassifier(max_features=1).fit(self.texts, self.labels)

        self.assertEqual(model.vocabulary_, ("hate",))
        self.assertEqual(set(model._term_class_counts), set(model.vocabulary_))

    def test_refit_discards_previous_training_state(self):
        model = EntropySentimentClassifier().fit(self.texts, self.labels)
        model.fit(["sunny and warm", "rainy and cold"], ["good", "bad"])

        self.assertEqual(model.classes_, ("bad", "good"))
        self.assertNotIn("love", model.information_gain_)

    def test_validates_training_inputs_and_parameters(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            EntropySentimentClassifier().fit(["good"], ["positive", "negative"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            EntropySentimentClassifier().fit([], [])
        with self.assertRaisesRegex(ValueError, "max_features"):
            EntropySentimentClassifier(max_features=0)
        with self.assertRaisesRegex(ValueError, "alpha"):
            EntropySentimentClassifier(alpha=0)
        with self.assertRaisesRegex(TypeError, "not strings"):
            EntropySentimentClassifier().fit("good", ["positive"])

    def test_requires_fitting_before_prediction(self):
        with self.assertRaisesRegex(RuntimeError, "fit"):
            EntropySentimentClassifier().predict(["good"])

    def test_tokenizer_is_case_insensitive_and_unicode_aware(self):
        self.assertEqual(tokenize("C'est ÉNORME!"), ["c'est", "énorme"])

    def test_tokenizer_marks_negation_scope(self):
        self.assertEqual(
            tokenize("This is not very good"),
            ["this", "is", "not", "not_very", "not_good"],
        )
        self.assertEqual(tokenize("wasn't helpful"), ["wasn't", "not_helpful"])

    def test_classifier_can_learn_simple_negated_sentiment(self):
        texts = [
            "good profit",
            "good revenue",
            "good gains",
            "not good profit",
            "not good revenue",
            "not good gains",
        ]
        model = EntropySentimentClassifier().fit(
            texts, ["positive"] * 3 + ["negative"] * 3
        )

        self.assertEqual(
            model.predict(["good outlook", "not good outlook"]),
            ["positive", "negative"],
        )

    def test_stratified_split_is_reproducible_and_keeps_each_class(self):
        labels = ["positive"] * 8 + ["negative"] * 8 + ["neutral"] * 8

        train_a, test_a = stratified_split(labels, test_size=0.25, seed=13)
        train_b, test_b = stratified_split(labels, test_size=0.25, seed=13)

        self.assertEqual((train_a, test_a), (train_b, test_b))
        self.assertEqual(len(test_a), 6)
        self.assertEqual({labels[index] for index in train_a}, set(labels))
        self.assertEqual({labels[index] for index in test_a}, set(labels))

    def test_metrics_and_csv_evaluation_beat_balanced_majority_baseline(self):
        positive = [
            f"good profit results {ending}"
            for ending in (
                "today", "reported", "rise", "grow",
                "strong", "excellent", "outlook", "investors",
            )
        ]
        negative = [
            f"bad loss results {ending}"
            for ending in (
                "today", "reported", "fall", "drop",
                "weak", "terrible", "outlook", "investors",
            )
        ]
        neutral = [
            f"company report {ending}"
            for ending in (
                "today", "published", "meeting", "schedule",
                "board", "shareholder", "filing", "quarterly",
            )
        ]
        texts = positive + negative + neutral
        labels = ["positive"] * len(positive) + ["negative"] * len(negative) + ["neutral"] * len(neutral)

        result = evaluate(texts, labels, test_size=1 / 3, seed=4)
        perfect = classification_metrics(
            ["good", "bad"], ["good", "bad"], ["bad", "good"]
        )

        self.assertEqual(perfect["accuracy"], 1.0)
        self.assertEqual(perfect["macro_f1"], 1.0)
        self.assertEqual(
            perfect["confusion_matrix"],
            {"bad": {"bad": 1, "good": 0}, "good": {"bad": 0, "good": 1}},
        )
        self.assertEqual(result["classifier"]["accuracy"], 1.0)
        self.assertEqual(result["classifier"]["macro_f1"], 1.0)
        self.assertGreater(
            result["classifier"]["macro_f1"],
            result["majority_baseline"]["macro_f1"],
        )
        self.assertEqual(result["test_examples"], 9)

    def test_csv_loader_checks_requested_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.csv"
            path.write_text(
                "review,sentiment\n\"Très bon produit\",positive\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_csv(path, "review", "sentiment"),
                (["Très bon produit"], ["positive"]),
            )
            with self.assertRaisesRegex(ValueError, "must contain columns"):
                load_csv(path, "text", "label")


if __name__ == "__main__":
    unittest.main()
