import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

class FrugalTextClassifier:
    def __init__(self, top_k_features: int = 5000):
        self.top_k = top_k_features
        self.vocabulary = {}
        self.classifier = LogisticRegression(max_iter=1000, solver='liblinear')

    def build_vocabulary(self, entropy_weighter):
        """Sélectionne les mots ayant le plus haut gain d'information."""
        terms_ig = {
            term: entropy_weighter.compute_information_gain(term)
            for term in entropy_weighter.term_frequencies.keys()
        }
        sorted_terms = sorted(terms_ig.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms[:self.top_k])}

    def vectorize(self, texts: List[str]) -> csr_matrix:
        """Création d'une matrice creuse ultra-légère."""
        rows, cols, data = [], [], []
        for row_idx, text in enumerate(texts):
            for word in set(text.split()):
                if word in self.vocabulary:
                    rows.append(row_idx)
                    cols.append(self.vocabulary[word])
                    data.append(1) # Ou le poids TF-IDF / Entropie
        return csr_matrix((data, (rows, cols)), shape=(len(texts), self.top_k))

    def train(self, texts: List[str], labels: List[int]):
        X = self.vectorize(texts)
        self.classifier.fit(X, labels)

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorize(texts)
        return self.classifier.predict(X)
