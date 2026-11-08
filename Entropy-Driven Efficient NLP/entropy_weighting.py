import math
from collections import defaultdict
from typing import List, Dict

class EntropyFeatureWeighter:
    def __init__(self):
        self.term_frequencies = defaultdict(int)
        self.class_frequencies = defaultdict(int)
        self.term_class_frequencies = defaultdict(lambda: defaultdict(int))
        self.total_documents = 0

    def fit(self, texts: List[str], labels: List[int]):
        """Calcule les distributions de probabilités."""
        for text, label in zip(texts, labels):
            self.total_documents += 1
            self.class_frequencies[label] += 1
            words = set(text.split()) # Présence binaire pour l'entropie
            for word in words:
                self.term_frequencies[word] += 1
                self.term_class_frequencies[word][label] += 1

    def compute_information_gain(self, term: str) -> float:
        """Calcule le gain d'information d'un terme spécifique."""
        p_t = self.term_frequencies[term] / self.total_documents
        if p_t == 0 or p_t == 1:
            return 0.0

        entropy_before = self._compute_entropy(self.class_frequencies.values(), self.total_documents)
        
        # Entropie conditionnelle
        entropy_if_present = self._compute_entropy(
            self.term_class_frequencies[term].values(), self.term_frequencies[term]
        )
        
        # Gain d'information
        return entropy_before - (p_t * entropy_if_present)

    def _compute_entropy(self, counts, total) -> float:
        entropy = 0.0
        for count in counts:
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
