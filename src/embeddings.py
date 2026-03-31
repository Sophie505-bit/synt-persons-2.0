from typing import List

_model_cache = {}


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name

    def _get_model(self):
        if self.model_name not in _model_cache:
            from sentence_transformers import SentenceTransformer
            _model_cache[self.model_name] = SentenceTransformer(self.model_name)
        return _model_cache[self.model_name]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
