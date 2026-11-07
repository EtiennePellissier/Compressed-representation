import gzip
# Remarque : Vous importerez ici les classes de votre fichier COMPRESSION
# from COMPRESSION import LempelZivCompressor, CustomEntropyEncoder

class CompressionDistance:
    def __init__(self, compressor_type: str = 'gzip'):
        self.compressor_type = compressor_type

    def get_compressed_size(self, text: bytes) -> int:
        """Interface avec votre module COMPRESSION ou gzip par défaut."""
        if self.compressor_type == 'gzip':
            return len(gzip.compress(text))
        # elif self.compressor_type == 'custom':
        #     return len(LempelZivCompressor.compress(text))
        raise ValueError("Compresseur non reconnu.")

    def compute_ncd(self, text1: str, text2: str) -> float:
        """
        Normalized Compression Distance (NCD).
        NCD(x, y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))
        """
        b1, b2 = text1.encode('utf-8'), text2.encode('utf-8')
        c1 = self.get_compressed_size(b1)
        c2 = self.get_compressed_size(b2)
        c12 = self.get_compressed_size(b1 + b2)
        
        return (c12 - min(c1, c2)) / max(c1, c2)
