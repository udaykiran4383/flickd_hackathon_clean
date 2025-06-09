import os
import json
import logging
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        return None

def main():
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Load catalog
    catalog_path = "data/catalog.csv"
    catalog_df = pd.read_csv(catalog_path)
    
    # Verify required columns exist
    required_columns = ['id', 'title', 'product_type', 'product_tags']
    missing_columns = [col for col in required_columns if col not in catalog_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in catalog: {missing_columns}")
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Process each product
    embeddings = []
    metadata = []
    
    for _, row in tqdm(catalog_df.iterrows(), total=len(catalog_df)):
        try:
            # Skip if no image URL is available
            if 'image_url' not in row or pd.isna(row['image_url']):
                logger.warning(f"No image URL for product {row['id']}")
                continue
                
            # Download image
            image = download_image(row['image_url'])
            if image is None:
                continue
                
            # Get CLIP embedding
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy()
                
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
                
            # Store embedding and metadata
            embeddings.append(embedding[0])  # Remove batch dimension
            metadata.append({
                'product_id': row['id'],  # Use 'id' instead of 'product_id'
                'title': row['title'],
                'product_type': row['product_type'],
                'product_tags': row['product_tags'],
                'product_collections': row.get('product_collections', '')  # Optional field
            })
            
        except Exception as e:
            logger.error(f"Error processing product {row['id']}: {str(e)}")
            continue
    
    if not embeddings:
        raise ValueError("No valid embeddings were generated. Check the catalog data and image URLs.")
    
    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)
    
    # Save embeddings as numpy array
    np.save('data/product_embeddings.npy', embeddings)
    
    # Save metadata
    with open('data/product_metadata.json', 'w') as f:
        json.dump(metadata, f)
        
    logger.info(f"Saved {len(embeddings)} product embeddings")
    logger.info(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main() 