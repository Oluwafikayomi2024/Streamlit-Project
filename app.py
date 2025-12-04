# app.py (FULL, FINAL CODE)

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any

# CRITICAL: These helper files MUST be present in the same directory
from data_prep import load_cleaned_data, get_unique_products
from vector_db import VectorDB_Simulator
from cnn_model import CNNModel_Service 

import tensorflow as tf 
import numpy as np 

# --------------------------------------------------------------------
# 1. INITIALIZATION
# --------------------------------------------------------------------

app = FastAPI(
    title="E-commerce Recommendation and Detection Service",
    description="A service for product recommendation via text, OCR, and image detection."
)

vector_db_service: VectorDB_Simulator = None
cnn_model_service: CNNModel_Service = None 

# --------------------------------------------------------------------
# 2. MODELS (Pydantic Schemas)
# --------------------------------------------------------------------

class ProductMatch(BaseModel):
    StockCode: str
    Description: str
    SimilarityScore: float = None

class RecommendationResponse(BaseModel):
    natural_language_response: str
    product_matches_array: List[ProductMatch]

class OCRDetectionResponse(BaseModel):
    extracted_text: str
    recommendation: RecommendationResponse

class ImageDetectionResponse(BaseModel):
    cnn_model_class: str 
    product_description: str
    recommendation: RecommendationResponse

# --------------------------------------------------------------------
# 3. HELPER FUNCTION
# --------------------------------------------------------------------

def get_product_recommendations(query: str) -> RecommendationResponse:
    """Queries the VectorDB and formats the results."""
    
    if not vector_db_service:
        raise HTTPException(status_code=500, detail="VectorDB service is not initialized.")
        
    top_matches = vector_db_service.query(query, top_k=5)
    
    if not top_matches:
        nl_response = f"I could not find a match for '{query}'."
    else:
        best_match_desc = top_matches[0]['Description']
        nl_response = (
            f"The closest match I found for '{query}' is: '{best_match_desc}'. "
            "Here are the top 5 most similar products based on the description."
        )

    return RecommendationResponse(
        natural_language_response=nl_response,
        product_matches_array=[
            ProductMatch(**match) for match in top_matches
        ]
    )

# --------------------------------------------------------------------
# 4. STARTUP EVENT
# --------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Initializes the services on application startup."""
    global vector_db_service, cnn_model_service
    
    try:
        # NOTE: This uses the corrected filename 'cleaned_data.csv'
        df = load_cleaned_data() 
        unique_products = get_unique_products(df)
    except Exception as e:
        raise RuntimeError(f"Failed to load data for services (Data Prep Error): {e}")

    if not unique_products.empty:
        product_descriptions = unique_products['Description'].tolist()
        
        vector_db_service = VectorDB_Simulator(unique_products)
        
        # NOTE: This loads your trained Keras model
        cnn_model_service = CNNModel_Service(
            model_path='cnn_product_detection_model.keras', 
            product_descriptions=product_descriptions
        )
    else:
        raise RuntimeError("Failed to load unique product data, cannot initialize services.")

# --------------------------------------------------------------------
# 5. ENDPOINTS
# --------------------------------------------------------------------

@app.get("/api/recommend/text", response_model=RecommendationResponse, tags=["Module 1"])
def recommend_by_text(query: str):
    return get_product_recommendations(query)

@app.post("/api/recommend/ocr", response_model=OCRDetectionResponse, tags=["Module 2"])
async def recommend_by_ocr(file: UploadFile = File(...)):
    simulated_extracted_text = "KNITTED UNION FLAG HOT WATER BOTTLE and a T-LIGHT HOLDER"
    recommendation_response = get_product_recommendations(simulated_extracted_text)
    
    return OCRDetectionResponse(
        extracted_text=simulated_extracted_text,
        recommendation=recommendation_response
    )

@app.post("/api/detect/image", response_model=ImageDetectionResponse, tags=["Module 3"])
async def detect_product_image(file: UploadFile = File(...)):
    
    if not cnn_model_service or not vector_db_service:
        raise HTTPException(status_code=500, detail="Core services are not initialized.")

    image_bytes = await file.read()
    predicted_class_name = cnn_model_service.detect_product(image_bytes)
    
    if predicted_class_name in ["PROCESSING_ERROR", "UNKNOWN_PRODUCT_INDEX"]:
        recommendation = RecommendationResponse(
            natural_language_response="Product identification failed due to an error or poor prediction.",
            product_matches_array=[]
        )
        return ImageDetectionResponse(
            cnn_model_class=predicted_class_name,
            product_description="Could not reliably identify product from image.",
            recommendation=recommendation
        )
        
    recommendation_response = get_product_recommendations(predicted_class_name)

    return ImageDetectionResponse(
        cnn_model_class=predicted_class_name,
        product_description=predicted_class_name,
        recommendation=recommendation_response
    )