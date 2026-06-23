import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


class LexicalSearcher:
    def __init__(self, metadata, params=None):
        self.vectorizer = None
        self.corpus_matrix = None
        self.metadata = metadata or []

        if params is not None:
            self._load_tfidf_or_fit(params)

    def _load_tfidf_or_fit(self, params):
        vec_path = Path(params["faiss"]["tfidf_vectorizer_path"])
        mat_path = Path(params["faiss"]["tfidf_matrix_path"])

        if vec_path.exists() and mat_path.exists():
            self.vectorizer = joblib.load(str(vec_path))
            self.corpus_matrix = joblib.load(str(mat_path))
            print(f"[Lexical] TF-IDF cargado ({len(self.metadata)} docs).")
        else:
            texts = [item["text"] for item in self.metadata]
            if texts:
                self.vectorizer = TfidfVectorizer(stop_words="english")
                self.corpus_matrix = self.vectorizer.fit_transform(texts)
                joblib.dump(self.vectorizer, vec_path)
                joblib.dump(self.corpus_matrix, mat_path)
                print(f"[Lexical] TF-IDF ajustado para {len(texts)} documentos (fallback).")

    def search(self, query, top_k=10):
        if self.corpus_matrix is None or not query.strip():
            return [], []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.corpus_matrix).flatten()
        top_indices = similarities.argsort()[::-1]

        results = []
        scores = []

        for idx in top_indices:
            if similarities[idx] == 0 or len(results) >= top_k:
                break
            results.append(self.metadata[idx])
            scores.append(float(similarities[idx]))

        return results, scores
