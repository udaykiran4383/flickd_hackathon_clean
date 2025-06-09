import torch
import torchvision.transforms as transforms
from PIL import Image
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import requests
from io import BytesIO
from urllib.parse import urlparse, parse_qs
import cv2
from sklearn.cluster import KMeans
import colorsys
import logging
import os
import json
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class ProductMatcher:
    # Similarity thresholds
    EXACT_MATCH_THRESHOLD = 0.8
    SIMILAR_MATCH_THRESHOLD = 0.6
    
    # Image size variations to try
    IMAGE_SIZES = ['512x', '800x', '1600x', '2048x', 'master']

    def __init__(self, embeddings_path: str, debug: bool = False):
        """Initialize product matcher with pre-computed embeddings."""
        self.debug = debug
        try:
            # Initialize CLIP model and processor
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            
            # Load product embeddings
            self.product_embeddings = np.load(embeddings_path)
            
            # Ensure embeddings are normalized
            self.product_embeddings = self.product_embeddings / np.linalg.norm(self.product_embeddings, axis=1, keepdims=True)
            
            # Load product metadata from catalog
            catalog_path = 'data/catalog.csv'
            self.product_metadata = pd.read_csv(catalog_path).to_dict('records')
            
            # Verify dimensions match
            if self.product_embeddings.shape[1] != 512:
                raise ValueError(f"Product embeddings dimension mismatch. Expected 512, got {self.product_embeddings.shape[1]}")
            
            # Initialize FAISS index with L2 distance
            self.index = faiss.IndexFlatL2(self.product_embeddings.shape[1])
            self.index.add(self.product_embeddings)
            
            if self.debug:
                logger.info(f"Loaded {len(self.product_metadata)} product embeddings")
                logger.info(f"Embedding dimension: {self.product_embeddings.shape[1]}")
                logger.info(f"CLIP model loaded on {self.device}")
                
        except Exception as e:
            logger.error(f"Error initializing ProductMatcher: {str(e)}")
            raise

    def _get_image_url_variations(self, url: str) -> List[str]:
        """Generate variations of the image URL with different sizes."""
        variations = []
        base_url = re.sub(r'_\d+x\.', '_', url)
        base_url = re.sub(r'\.jpg\?v=\d+', '.jpg', base_url)
        
        for size in self.IMAGE_SIZES:
            # Try with size in filename
            variations.append(base_url.replace('.jpg', f'_{size}.jpg'))
            # Try with size as parameter
            variations.append(f"{base_url}?width={size.replace('x', '')}")
            
        return variations

    def _download_image(self, url: str) -> Image.Image:
        """Download image from URL and return PIL Image."""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.warning(f"Error downloading image from {url}: {str(e)}")
            raise

    def _load_or_create_index(self) -> np.ndarray:
        """Load existing FAISS index or create new one."""
        try:
            if self.index_path.exists():
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(str(self.index_path))
                return np.load(self.cache_dir / "catalog_embeddings.npy")
            
            logger.info("Creating new FAISS index...")
            embeddings = []
            valid_products = []
            
            # Process each catalog image
            for idx, row in self.merged_df.iterrows():
                try:
                    if pd.isna(row['image_url']):
                        continue
                        
                    # Download and process image
                    image = self._download_image(row['image_url'])
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    
                    # Get image embedding
                    with torch.no_grad():
                        image_features = self.model.get_image_features(**inputs)
                        embedding = image_features.cpu().numpy()
                        embeddings.append(embedding)
                        valid_products.append(row)
                        
                    if idx % 100 == 0:
                        logger.info(f"Processed {idx} images...")
                        
                except Exception as e:
                    continue  # Skip problematic images
            
            if not embeddings:
                raise ValueError("No valid images found in catalog")
            
            # Create FAISS index
            embeddings = np.vstack(embeddings)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            # Update merged_df to only include valid products
            self.merged_df = pd.DataFrame(valid_products)
            
            # Save index and embeddings
            faiss.write_index(self.index, str(self.index_path))
            np.save(self.cache_dir / "catalog_embeddings.npy", embeddings)
            
            logger.info(f"Created FAISS index with {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise

    def _get_match_type(self, similarity: float) -> str:
        """Get match type based on similarity score."""
        if similarity >= self.EXACT_MATCH_THRESHOLD:
            return "Exact Match"
        elif similarity >= self.SIMILAR_MATCH_THRESHOLD:
            return "Similar Match"
        else:
            return "No Match"

    def _extract_clothing_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract clothing region from person detection."""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        
        # Focus on upper body (top 60% of person)
        upper_body_height = int(height * 0.6)
        upper_body_y2 = y1 + upper_body_height
        
        # Extract upper body region
        upper_body = frame[y1:upper_body_y2, x1:x2]
        return upper_body

    def match_products(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match detected objects to products using CLIP and FAISS."""
        try:
            matched_products = []
            
            for obj in objects:
                try:
                    # Handle person detections
                    if obj['type'] == 'person':
                        # Extract clothing region from person
                        clothing_region = self._extract_clothing_region(obj['frame'], obj['bbox'])
                        object_image = Image.fromarray(cv2.cvtColor(clothing_region, cv2.COLOR_BGR2RGB))
                    else:
                        # For other objects, use the full bbox
                        x1, y1, x2, y2 = obj['bbox']
                        object_image = Image.fromarray(cv2.cvtColor(obj['frame'][y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                    
                    # Get CLIP embedding
                    inputs = self.processor(images=object_image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        object_features = self.model.get_image_features(**inputs)
                        object_embedding = object_features.cpu().numpy()
                        # Normalize embedding
                        object_embedding = object_embedding / np.linalg.norm(object_embedding)
                    
                    # Search in FAISS index for top 10 matches (increased from 5)
                    k = 10
                    distances, indices = self.index.search(object_embedding, k)
                    
                    # Process each match
                    for i in range(k):
                        # Calculate similarity score (convert L2 distance to similarity)
                        similarity = 1 / (1 + distances[0][i])
                        
                        # Get match type
                        match_type = self._get_match_type(similarity)
                        
                        # Only include matches above threshold
                        if match_type != "No Match":
                            product_idx = indices[0][i]
                            product = self.product_metadata[product_idx]
                            
                            # Extract color from product tags
                            color = "unknown"
                            if 'product_tags' in product and pd.notna(product['product_tags']):
                                tags = product['product_tags'].split(', ')
                                for tag in tags:
                                    if tag.startswith('Colour:'):
                                        color = tag.split(':')[1]
                                        break
                            
                            # Add match details
                            product.update({
                                'match_type': match_type,
                                'similarity': float(similarity),
                                'detection_confidence': obj['confidence'],
                                'frame_number': obj['frame_number'],
                                'detected_type': obj['type'],
                                'color': color
                            })
                            
                            matched_products.append(product)
                            logger.info(f"Matched {obj['type']} to {product['product_type']} with {match_type} (similarity: {similarity:.3f})")
                    
                except Exception as e:
                    logger.warning(f"Error matching object: {str(e)}")
                    continue
            
            logger.info(f"Matched {len(matched_products)} products")
            return matched_products
            
        except Exception as e:
            logger.error(f"Error matching products: {str(e)}")
            return []

    def _compute_catalog_embeddings(self) -> np.ndarray:
        """Compute embeddings for all catalog images."""
        embeddings = []
        total_images = len(self.catalog_df)
        successful_images = 0
        
        for idx, row in self.catalog_df.iterrows():
            try:
                logger.info(f"Processing image {idx + 1}/{total_images}: {row['image_url']}")
                
                # Download and load image from URL
                response = requests.get(row['image_url'], timeout=10)  # Added timeout
                if response.status_code != 200:
                    logger.error(f"Failed to download image from {row['image_url']}: HTTP {response.status_code}")
                    continue
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    logger.error(f"Invalid content type for {row['image_url']}: {content_type}")
                    continue
                
                # Try to open the image
                try:
                    img = Image.open(BytesIO(response.content))
                    # Verify the image was loaded correctly
                    img.verify()
                    # Reopen the image since verify() closes it
                    img = Image.open(BytesIO(response.content))
                    
                    # Check image dimensions
                    if img.size[0] < 10 or img.size[1] < 10:
                        logger.error(f"Image too small: {img.size} for {row['image_url']}")
                        continue
                        
                except Exception as img_error:
                    logger.error(f"Failed to open image from {row['image_url']}: {str(img_error)}")
                    continue
                
                img_tensor = self.transform(img).unsqueeze(0)
                
                # Compute embedding
                with torch.no_grad():
                    embedding = self.model(img_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    successful_images += 1
                    
            except Exception as e:
                logger.error(f"Error computing embedding for {row['image_url']}: {e}")
                continue
        
        logger.info(f"Successfully processed {successful_images}/{total_images} images")
        if successful_images == 0:
            raise RuntimeError("No images were successfully processed")
            
        return np.array(embeddings)
    
    def match_product(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Match a product image against the catalog."""
        try:
            # Convert numpy array to PIL Image
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Transform image
            img_tensor = self.transform(img).unsqueeze(0)
            
            # Compute embedding
            with torch.no_grad():
                embedding = self.model(img_tensor)
                embedding = embedding.cpu().numpy().flatten()
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
            
            # Compute cosine similarity with catalog
            similarities = np.dot(self.product_embeddings, embedding)
            
            # Get top matches
            top_k = 5
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            matches = []
            for idx in top_indices:
                if similarities[idx] > 0.5:  # Only include high confidence matches
                    match = {
                        'product_id': self.product_metadata[idx]['product_id'],
                        'confidence': float(similarities[idx])
                    }
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error matching product: {e}")
            return []
    
    def detect_color(self, image: np.ndarray) -> Tuple[str, float]:
        """Detect dominant colors in an image using K-means clustering."""
        try:
            # Reshape image for clustering
            pixels = image.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(pixels)
            
            # Get colors and their percentages
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            color_percentages = np.bincount(labels) / len(labels)
            
            # Sort colors by percentage
            color_info = sorted(zip(colors, color_percentages), key=lambda x: x[1], reverse=True)
            
            # Convert RGB to HSV for better color naming
            hsv_colors = []
            for rgb, percentage in color_info:
                h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
                hsv_colors.append((h*360, s*100, v*100, percentage))
            
            # Define color ranges
            color_ranges = {
                'red': [(0, 10), (350, 360)],
                'orange': [(10, 30)],
                'yellow': [(30, 60)],
                'green': [(60, 150)],
                'blue': [(150, 240)],
                'purple': [(240, 300)],
                'pink': [(300, 350)],
                'white': [(0, 360, 0, 20, 80, 100)],
                'black': [(0, 360, 0, 100, 0, 20)],
                'gray': [(0, 360, 0, 20, 20, 80)]
            }
            
            # Get color names
            colors = []
            for h, s, v, percentage in hsv_colors:
                color_name = 'unknown'
                if v < 20:  # Dark
                    color_name = 'black'
                elif v > 80 and s < 20:  # Light
                    color_name = 'white'
                elif s < 20:  # Gray
                    color_name = 'gray'
                else:
                    for name, ranges in color_ranges.items():
                        if any(low <= h <= high for low, high in ranges):
                            color_name = name
                            break
                colors.append((color_name, percentage))
            
            return colors
            
        except Exception as e:
            logger.error(f"Error detecting colors: {e}")
            return [('unknown', 1.0)]
            
    def load_catalog(self, catalog_path: str):
        """Load the product catalog and compute embeddings."""
        try:
            # Check if we have cached embeddings
            if self.embeddings_cache.exists() and self.catalog_cache.exists():
                logger.info("Loading cached catalog and embeddings...")
                catalog_data = json.loads(self.catalog_cache.read_text())
                self.catalog_df = pd.DataFrame(catalog_data)  # Convert back to DataFrame
                self.product_embeddings = np.array(self.catalog_df['embedding'].tolist())
                logger.info(f"Loaded {len(self.catalog_df)} cached product embeddings")
                return
            
            # Load product catalog
            self.catalog_df = pd.read_csv(catalog_path)
            logger.info(f"Loaded catalog with {len(self.catalog_df)} images")
            
            # Compute embeddings
            self.product_embeddings = self._compute_catalog_embeddings()
            logger.info("Catalog embeddings computed successfully")
            
            # Cache the results
            np.save(self.embeddings_cache, self.product_embeddings)
            self.catalog_cache.write_text(json.dumps(self.catalog_df.to_dict('records')))
            
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            raise
            
    def _select_best_image_url(self, urls: List[str]) -> str:
        """Select the best quality image URL from a list of URLs."""
        # Sort URLs by quality indicators
        quality_scores = []
        for url in urls:
            score = 0
            # Prefer larger images
            if '1600x' in url:
                score += 3
            elif '512x' in url:
                score += 1
            # Prefer original quality
            if 'quality=60' not in url:
                score += 2
            # Prefer main product images
            if any(x in url.lower() for x in ['main', 'front', '1_', 'web1']):
                score += 2
            quality_scores.append((score, url))
        
        # Return URL with highest score
        return max(quality_scores, key=lambda x: x[0])[1]
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for CLIP model."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def crop_and_match(self, frame: np.ndarray, bbox: List[float], threshold: float = 0.75) -> List[Dict]:
        """Crop image using bbox and match against catalog."""
        try:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within image bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Crop image
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return []
                
            # Match product
            return self.match_product(cropped)
            
        except Exception as e:
            logger.error(f"Error in crop_and_match: {e}")
            return []

    def find_matches(self, query_embedding: np.ndarray, object_type: str, 
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matching products for a detected object."""
        try:
            # Ensure query embedding is normalized
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Verify dimensions match
            if query_embedding.shape[1] != self.product_embeddings.shape[1]:
                raise ValueError(f"Query embedding dimension mismatch. Expected {self.product_embeddings.shape[1]}, got {query_embedding.shape[1]}")
            
            # Search for similar products using L2 distance
            distances, indices = self.index.search(query_embedding, top_k)
            
            if self.debug:
                logger.info(f"Search results - distances: {distances}, indices: {indices}")
            
            matches = []
            for dist, idx in zip(distances[0], indices[0]):
                # Convert L2 distance to similarity score (1 / (1 + distance))
                similarity = 1 / (1 + dist)
                
                # Skip if similarity is too low
                if similarity < self.SIMILAR_MATCH_THRESHOLD:
                    continue
                    
                # Get product metadata
                if idx >= len(self.product_metadata):
                    logger.warning(f"Index {idx} out of bounds for product metadata")
                    continue
                    
                product = self.product_metadata[idx]
                
                # Skip if product type doesn't match (if category exists)
                if 'category' in product and product['category'] != object_type:
                    continue
                
                # Determine match type
                match_type = "exact" if similarity >= self.EXACT_MATCH_THRESHOLD else "similar"
                
                # Get color from product tags if available
                color = "unknown"
                if 'product_tags' in product and pd.notna(product['product_tags']):
                    tags = product['product_tags'].split(', ')
                    for tag in tags:
                        if tag.startswith('Colour:'):
                            color = tag.split(':')[1]
                            break
                
                matches.append({
                    "type": object_type,  # Use detected object type
                    "color": color,
                    "matched_product_id": product.get('product_id', 'unknown'),
                    "match_type": match_type,
                    "confidence": float(similarity)
                })
            
            if self.debug:
                logger.info(f"Found {len(matches)} matches for {object_type}")
                
            return matches
            
        except Exception as e:
            logger.error(f"Error finding product matches: {str(e)}")
            if self.debug:
                logger.error(f"Query embedding shape: {query_embedding.shape}")
                logger.error(f"Product embeddings shape: {self.product_embeddings.shape}")
                logger.error(f"Product metadata length: {len(self.product_metadata)}")
                logger.error(f"Index dimension: {self.index.d}")
            return [] 