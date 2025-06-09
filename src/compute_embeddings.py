import torch
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.warning(f"Error downloading image from {url}: {str(e)}")
        raise

def compute_embeddings(catalog_path: str, output_path: str):
    """Compute CLIP embeddings for all catalog images."""
    try:
        # Initialize CLIP model and processor
        logger.info("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info(f"CLIP model loaded on {device}")
        
        # Load catalog
        logger.info("Loading catalog...")
        catalog_df = pd.read_csv(catalog_path)
        logger.info(f"Loaded {len(catalog_df)} products")
        
        # Compute embeddings
        embeddings = []
        valid_products = []
        
        logger.info("Computing embeddings...")
        for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df)):
            try:
                if pd.isna(row['image_url']):
                    continue
                    
                # Download and process image
                image = download_image(row['image_url'])
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # Get image embedding
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    embedding = image_features.cpu().numpy()
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    valid_products.append(row)
                    
            except Exception as e:
                logger.warning(f"Error processing image {idx}: {str(e)}")
                continue
        
        if not embeddings:
            raise ValueError("No valid images found in catalog")
        
        # Stack embeddings
        embeddings = np.vstack(embeddings)
        
        # Save embeddings
        logger.info(f"Saving {len(embeddings)} embeddings to {output_path}")
        np.save(output_path, embeddings)
        
        # Save updated catalog
        updated_catalog_path = os.path.join(os.path.dirname(output_path), 'updated_catalog.csv')
        pd.DataFrame(valid_products).to_csv(updated_catalog_path, index=False)
        logger.info(f"Saved updated catalog to {updated_catalog_path}")
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    compute_embeddings('data/catalog.csv', 'data/product_embeddings.npy') 