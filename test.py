# DIAGNOSTIC CODE: Run this to find the failing import line

print("--- Starting Import Check ---")

try:
    import pandas as pd
    print("1. pandas imported successfully.")
except Exception as e:
    print(f"FAILED: pandas import failed with error: {e}")

try:
    import tensorflow as tf
    print("2. tensorflow imported successfully.")
except Exception as e:
    print(f"FAILED: tensorflow import failed with error: {e}")

try:
    from data_prep import load_cleaned_data, get_unique_products
    print("3. data_prep imported successfully.")
except Exception as e:
    print(f"FAILED: data_prep import failed with error: {e}")

try:
    from vector_db import VectorDB_Simulator
    print("4. vector_db imported successfully.")
except Exception as e:
    print(f"FAILED: vector_db import failed with error: {e}")

try:
    from cnn_model import CNNModel_Service
    print("5. cnn_model imported successfully.")
except Exception as e:
    print(f"FAILED: cnn_model import failed with error: {e}")
    
print("--- Import Check Complete ---")




import os
print("Current Working Directory (CWD):")
print(os.getcwd())
print("\nFiles in CWD:")
print(os.listdir(os.getcwd()))