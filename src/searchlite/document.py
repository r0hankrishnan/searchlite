from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Document():
    def __init__(self, texts, metadata, model = None):
        self.texts = texts
        self.metadata = metadata
        self.embeddings = None
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
   
    def embed(self):
        if self.embeddings is not None:
            return "Already embedded texts"
        else:
            embeddings = self.model.encode(self.texts)
            self.embeddings = embeddings
    
    def query(self, query_text, top_k = 3):
        if self.embeddings is not None:
            raise ValueError("Text embeddings not found. Please run .embed() before querying.")
        else:
            query_embedding = self.model.encode(query_text)
            similarities = cosine_similarity(X = self.embeddings, Y = query_embedding)

    