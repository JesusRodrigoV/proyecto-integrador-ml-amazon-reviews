import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class LexicalSearcher:
    def __init__(self, id_map_path):
        self.id_map_path = Path(id_map_path)
        # Usamos stop_words='english' para ignorar palabras comunes como 'the', 'and', 'is'
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.corpus_matrix = None
        self.metadata = []
        self._load_and_fit()

    def _load_and_fit(self):
        """Carga el id_map y ajusta el modelo TF-IDF con los textos"""
        if not self.id_map_path.exists():
            print(f"[Lexical] Advertencia: No se encontro {self.id_map_path}")
            return
        
        with open(self.id_map_path, "rb") as f:
            self.metadata = pickle.load(f)

        texts = [item["text"] for item in self.metadata]
        
        if texts:
            self.corpus_matrix = self.vectorizer.fit_transform(texts)
            print(f"[Lexical] TF-IDF ajustado para {len(texts)} documentos.")

    def search(self, query, top_k=10):
        """Busca la consulta usando coincidencia de palabras exactas (TF-IDF)"""
        if self.corpus_matrix is None or not query.strip():
            return [], []

        # Transformar la consulta al mismo espacio vectorial
        query_vec = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vec, self.corpus_matrix).flatten()

        # Obtener los índices de mayor a menor similitud
        top_indices = similarities.argsort()[::-1]

        results = []
        scores = []
        
        for idx in top_indices:
            # Si la similitud es 0, significa que la consulta no comparte ninguna palabra con el texto
            if similarities[idx] == 0 or len(results) >= top_k:
                break
            results.append(self.metadata[idx])
            scores.append(float(similarities[idx]))

        return results, scores