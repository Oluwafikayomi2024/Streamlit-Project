import numpy as np

class VectorDB_Simulator:
    def __init__(self, products, embeddings):
        self.products = products
        self.embeddings = embeddings

    def search(self, query_embedding, top_k=5):
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.products[i] for i in top_indices]
