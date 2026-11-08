import time
import tracemalloc
from data_ingestion import DataLoader
from entropy_weighting import EntropyFeatureWeighter
from efficient_classifier import FrugalTextClassifier

def main():
    print("Démarrage du profilage de l'architecture...")
    tracemalloc.start()
    start_time = time.time()

    # 1. Ingestion
    loader = DataLoader("dataset.csv")
    texts, labels = [], []
    for text, label in loader.stream_data():
        texts.append(text)
        labels.append(label)

    # 2. Théorie de l'information (Entropie)
    weighter = EntropyFeatureWeighter()
    weighter.fit(texts, labels)

    # 3. Classification Frugale
    model = FrugalTextClassifier(top_k_features=2000)
    model.build_vocabulary(weighter)
    model.train(texts, labels)

    # 4. Métriques et Profilage
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.time()
    
    print(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")
    print(f"Pic de mémoire vive (RAM) : {peak / 10**6:.2f} MB")
    tracemalloc.stop()

if __name__ == "__main__":
    main()
