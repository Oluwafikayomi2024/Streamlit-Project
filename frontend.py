# frontend.py
import streamlit as st
import requests
import io
from PIL import Image
import pandas as pd
# --- CONFIGURATION ---
# Base URL for your FastAPI server (Ensure your FastAPI server is running)
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="E-commerce Product Search & Recommendation",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def display_recommendations(api_response):
    """Displays the natural language response and the product table."""
    try:
        # Check if the response structure is consistent with the standard RecommendationResponse
        recommendation = api_response.get('recommendation', api_response)
        
        # 1. Natural Language Response
        st.subheader("💡 Recommended Solution")
        st.markdown(f"**{recommendation['natural_language_response']}**")

        # 2. Product Matches Table
        matches = recommendation['product_matches_array']
        if matches:
            st.subheader("📋 Top Product Matches")
            # Convert list of dicts to DataFrame for display
            df_matches = pd.DataFrame(matches)
            
            # Select and rename columns for a clean UI
            if 'StockCode' in df_matches.columns:
                df_matches = df_matches[['StockCode', 'Description', 'SimilarityScore']]
                df_matches.columns = ['Stock Code', 'Description', 'Similarity Score']
            
            st.dataframe(df_matches, use_container_width=True, hide_index=True)
        else:
            st.warning("No similar products found.")
            
    except Exception as e:
        st.error(f"Error processing API response: {e}")
        st.json(api_response) # Display raw JSON for debugging


# --- MAIN APP LAYOUT ---

st.title("🛍️ E-commerce Product Discovery System")

# Create tabs for each interface
tab1, tab2, tab3 = st.tabs([
    "1. Text Search (Vector DB)", 
    "2. Image Query (OCR)", 
    "3. Product Image Detection (CNN)"
])

# ==============================================================================
# FRONTEND PAGE 1: TEXT QUERY INTERFACE (Module 1, Endpoint 1)
# ==============================================================================

with tab1:
    st.header("Text Search: Find products using natural language.")
    
    query = st.text_input(
        "Enter your product query:", 
        placeholder="e.g., I need a cheap, vintage-style metal kitchen container"
    )

    if st.button("Search Products (Text)", type="primary"):
        if query:
            with st.spinner("Searching Vector Database..."):
                try:
                    response = requests.get(
                        f"{API_URL}/recommend/text",
                        params={"query": query}
                    )
                    response.raise_for_status() # Raise exception for bad status codes
                    api_response = response.json()
                    
                    display_recommendations(api_response)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API: {e}. Ensure your FastAPI server is running.")
        else:
            st.warning("Please enter a query.")

# ==============================================================================
# FRONTEND PAGE 2: IMAGE QUERY INTERFACE (Module 2, Endpoint 2)
# ==============================================================================

with tab2:
    st.header("Image Query: Search based on handwritten notes or images of text.")
    
    uploaded_file = st.file_uploader(
        "Upload an image containing text (e.g., a note or receipt):", 
        type=["png", "jpg", "jpeg"]
    )

    if st.button("Search Products (Image OCR)", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing image with OCR and searching..."):
                try:
                    # Prepare file for upload
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    response = requests.post(
                        f"{API_URL}/recommend/ocr",
                        files=files
                    )
                    response.raise_for_status()
                    api_response = response.json()

                    # Display the extracted text first
                    extracted_text = api_response.get('extracted_text', 'N/A')
                    st.info(f"**Extracted Text:** '{extracted_text}'")

                    # Display the search results
                    display_recommendations(api_response)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API: {e}. Ensure your FastAPI server is running.")
        else:
            st.warning("Please upload an image.")

# ==============================================================================
# FRONTEND PAGE 3: PRODUCT IMAGE UPLOAD INTERFACE (Module 3, Endpoint 3)
# ==============================================================================

with tab3:
    st.header("Product Image: Identify a product and find similar items.")
    
    uploaded_prod_file = st.file_uploader(
        "Upload an image of the product you want to identify:", 
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_prod_file is not None:
        st.image(uploaded_prod_file, caption='Uploaded Product Image', width=200)

    if st.button("Detect Product (CNN)", type="primary"):
        if uploaded_prod_file is not None:
            with st.spinner("Detecting product using CNN model and searching..."):
                try:
                    # Rewind file pointer before sending to API
                    uploaded_prod_file.seek(0)
                    
                    # Prepare file for upload
                    files = {'file': (uploaded_prod_file.name, uploaded_prod_file.getvalue(), uploaded_prod_file.type)}
                    
                    response = requests.post(
                        f"{API_URL}/detect/image",
                        files=files
                    )
                    response.raise_for_status()
                    api_response = response.json()

                    # 1. CNN Classification Result
                    cnn_class = api_response.get('cnn_model_class', 'N/A')
                    prod_desc = api_response.get('product_description', 'N/A')
                    
                    st.success(f"**CNN Identified Class:** `{cnn_class}`")
                    st.markdown(f"**Identified Product Description:** {prod_desc}")

                    # 2. Display the search results (uses the same recommendation structure)
                    display_recommendations(api_response)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API: {e}. Ensure your FastAPI server is running.")
        else:
            st.warning("Please upload a product image.")