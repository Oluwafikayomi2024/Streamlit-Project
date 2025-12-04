from services.vector_db_simulator import VectorDB_Simulator
from services.cnn_model_service import CNNModel_Service

# ---------------------------------------
# INITIALIZATION (your models + embeddings)
# ---------------------------------------

def load_vector_db():
    products = ["Product A", "Product B", "Product C"]
    embeddings = [
        [0.4, 0.2, 0.8],
        [0.1, 0.9, 0.3],
        [0.7, 0.1, 0.5]
    ]
    return VectorDB_Simulator(products, embeddings)

def load_cnn_model():
    class DummyModel:
        def __call__(self, x):
            import torch
            return torch.tensor([[0.1, 0.7, 0.2]])
    labels = ["Electronics", "Clothing", "Shoes"]
    return CNNModel_Service(DummyModel(), labels)

vector_db = load_vector_db()
cnn_model = load_cnn_model()

# ---------------------------------------
# TEXT QUERY INTERFACE
# ---------------------------------------

def text_query(query):
    print(f"Processing text query: {query}")

    query_embedding = [0.5, 0.3, 0.6]  # dummy example
    results = vector_db.search(query_embedding)
    
    print("Top recommendations:")
    for r in results:
        print(" -", r)

# ---------------------------------------
# IMAGE CLASSIFICATION INTERFACE
# ---------------------------------------

def classify_image(path):
    print(f"Classifying image: {path}")
    result = cnn_model.predict(path)
    print("Detected Category:", result)

# ---------------------------------------
# OCR → RECOMMENDATION (Optional)
# ---------------------------------------

def ocr_and_recommend(path):
    print(f"OCR processing image: {path}")
    extracted_text = "dummy text from OCR"  # plug in real OCR
    text_query(extracted_text)

# ---------------------------------------
# RUN DEMOS
# ---------------------------------------

if __name__ == "__main__":
    text_query("Show me gaming laptops")

    classify_image("example.jpg")

    ocr_and_recommend("handwritten.jpg")
