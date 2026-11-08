import pandas as pd
import re
from typing import Iterator, Tuple

class DataLoader:
    def __init__(self, filepath: str, chunk_size: int = 1000):
        self.filepath = filepath
        self.chunk_size = chunk_size

    def clean_text(self, text: str) -> str:
        """Nettoyage frugal par expressions régulières."""
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text) # Suppression HTML
        text = re.sub(r'[^a-z0-9\s]', '', text) # Ponctuation
        return text

    def stream_data(self) -> Iterator[Tuple[str, int]]:
        """Lecture par lots (chunks) pour préserver la mémoire vive."""
        for chunk in pd.read_csv(self.filepath, chunksize=self.chunk_size):
            for _, row in chunk.iterrows():
                yield self.clean_text(row['text']), row['label']
