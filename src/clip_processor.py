import logging
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

logger = logging.getLogger(__name__)

class FashionCLIPProcessor:
    def __init__(self, debug: bool = False):
        """Initialize CLIP model and processor."""
        self.debug = debug
        try:
            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            
            if self.debug:
                logger.info(f"CLIP model loaded on {self.device}")
                
        except Exception as e:
            logger.error(f"Error initializing CLIP model: {str(e)}")
            raise

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Get CLIP embedding for an image."""
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy
            embedding = image_features.cpu().numpy()
            
            if self.debug:
                logger.info(f"Generated CLIP embedding with shape {embedding.shape}")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating CLIP embedding: {str(e)}")
            return None 