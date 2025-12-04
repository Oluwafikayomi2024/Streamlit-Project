# data_prep.py (FINAL CORRECTED VERSION)

import pandas as pd
import numpy as np

# Load the cleaned dataset
def load_cleaned_data(file_path='cleaned_data.csv'):
    """Loads the cleaned e-commerce dataset and strips column whitespace."""
    try:
        df = pd.read_csv(file_path)
        
        # 💡 THE FIX: Strip whitespace from all column names
        df.columns = df.columns.str.strip() 
        
        # Ensure InvoiceDate is datetime if needed later
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please run the initial data cleaning step.")
        return pd.DataFrame()

# Generate a unique list of products for vectorization
def get_unique_products(df):
    """Returns a DataFrame of unique products and their descriptions."""
    # Group by StockCode and get the first non-null Description
    unique_products = df.groupby('StockCode')['Description'].first().reset_index()
    unique_products.columns = ['StockCode', 'Description']
    # Drop any rows where Description is missing after grouping
    unique_products.dropna(subset=['Description'], inplace=True)
    # Remove duplicates on Description itself, keeping the first occurrence
    unique_products.drop_duplicates(subset=['Description'], keep='first', inplace=True)
    
    # Clean the descriptions for use as class names/queries
    unique_products['Description'] = unique_products['Description'].astype(str).str.strip()
    
    return unique_products