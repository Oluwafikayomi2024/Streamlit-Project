# cnn_model.py
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np

class CNNModel_Service:
    """CNN Model Service handles loading and inference of the trained model."""
    def __init__(self, model_path='cnn_product_detection_model.keras', product_descriptions=None):
        # NOTE: This module requires tensorflow and the saved model file
        self.product_classes = product_descriptions if product_descriptions else []
        self.model = None
        self.IMG_SIZE = 150 
        
        try:
            # Load the trained model
            self.model = keras.models.load_model(model_path)
            print(f"CNN Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"WARNING: Could not load CNN model ('{model_path}'). Falling back to simulator. Error: {e}")
            if not self.product_classes:
                 self.product_classes = ["MOCK_PRODUCT_A", "MOCK_PRODUCT_B"]

    def detect_product(self, image_bytes: bytes):
        """Processes an image, runs CNN inference, and returns the predicted class name."""
        
        if self.model is None:
             # Simulation fallback if model failed to load
             return random.choice(self.product_classes)
        
        try:
            # --- 1. Load and Preprocess Image ---
            img_tensor = tf.io.decode_image(image_bytes, channels=3)
            img_tensor = tf.image.resize(img_tensor, [self.IMG_SIZE, self.IMG_SIZE])
            img_tensor = tf.expand_dims(img_tensor, axis=0) # Add batch dimension
            
            # --- 2. Inference ---
            predictions = self.model.predict(img_tensor, verbose=0)
            predicted_index = np.argmax(predictions[0])
            
            # --- 3. Get Class Name ---
            if predicted_index < len(self.product_classes):
                predicted_class_name = self.product_classes[predicted_index]
            else:
                predicted_class_name = "UNKNOWN_PRODUCT_INDEX"
            
            return predicted_class_name

        except Exception as e:
            print(f"Error during CNN detection: {e}")
            return "PROCESSING_ERROR"