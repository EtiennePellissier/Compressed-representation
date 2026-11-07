import unittest

from entropy_sentiment import EntropySentimentClassifier, tokenize


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


if __name__ == "__main__":
    unittest.main()
