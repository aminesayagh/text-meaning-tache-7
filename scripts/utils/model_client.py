import requests
import numpy as np
import json
from typing import List, Dict, Any
import logging

class ModelClient:
    def __init__(self, host: str = "tensorflow-serving", port: int = 8501):
        self.base_url = f"http://{host}:{port}/v1/models/text_classifier"
        self.logger = logging.getLogger(__name__)
        
    def predict(self, texts: List[str], version: str = "production") -> Dict[str, Any]:
        """
        Make predictions using the TensorFlow Serving endpoint
        """
        # Convert texts to model input format
        instances = [{"text": text} for text in texts]
        
        # Prepare request URL with version label
        url = f"{self.base_url}/labels/{version}:predict"
        
        try:
            response = requests.post(url, json={"instances": instances})
            response.raise_for_status()
            
            predictions = response.json()
            return predictions["predictions"]
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error making prediction request: {e}")
            raise
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get the status of the model serving
        """
        try:
            response = requests.get(f"{self.base_url}")
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting model status: {e}")
            raise
            
    def get_model_metadata(self, version: str = "production") -> Dict[str, Any]:
        """
        Get model metadata for specific version
        """
        try:
            response = requests.get(f"{self.base_url}/labels/{version}/metadata")
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting model metadata: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    client = ModelClient()
    
    # Test texts
    texts = [
        "Example text in English",
        "Exemple de texte en français",
        "مثال نص باللغة العربية"
    ]
    
    try:
        # Get model status
        status = client.get_model_status()
        print("Model Status:", json.dumps(status, indent=2))
        
        # Make predictions
        predictions = client.predict(texts)
        print("\nPredictions:", json.dumps(predictions, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")