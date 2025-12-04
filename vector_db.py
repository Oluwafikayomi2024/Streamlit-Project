# vector_db.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

class VectorDB_Simulator:
    """SIMULATED Pinecone Vector Database using TF-IDF and Cosine Similarity."""
    def __init__(self, df_products):
        # NOTE: This module requires scikit-learn
        self.products = df_products.reset_index(drop=True)
        self.descriptions = self.products['Description'].tolist()
        
        # Using TF-IDF to simulate text embeddings
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.product_vectors = self.vectorizer.fit_transform(self.descriptions)
        
        print("VectorDB Initialized using TF-IDF embeddings.")

    def embed_query(self, query_text: str):
        """Simulates creating an embedding for a natural language query."""
        # Using the same vectorizer to transform the query
        return self.vectorizer.transform([query_text])

    def query(self, query_text: str, top_k: int = 5):
        """Performs a similarity search using Cosine Similarity."""
        
        try:
            query_vector = self.embed_query(query_text)
        except:
            return []
            
        # Check if the query vector is all zeros 
        if query_vector.sum() == 0:
            return random.sample(self.products.to_dict('records'), min(top_k, len(self.products)))
            
        similarities = cosine_similarity(query_vector, self.product_vectors).flatten()

        # Get the indices of the top_k most similar products
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Compile results
        results = []
        for i in top_indices:
            product_info = self.products.iloc[i].to_dict()
            product_info['SimilarityScore'] = round(float(similarities[i]), 4) # Ensure compatibility
            results.append(product_info)
            
        return results