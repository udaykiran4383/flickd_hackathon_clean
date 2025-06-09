import json
from typing import Dict, Any, List, Optional
from .object_detector import ObjectDetector
from .product_matcher import ProductMatcher
from .vibe_classifier import VibeClassifier
import numpy as np
import os
import logging
import cv2
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlickdPipeline:
    def __init__(self, vibes_list_path: str, catalog_path: str, debug: bool = False):
        """Initialize the Flickd pipeline with required components."""
        self.debug = debug
        if self.debug:
            logger.info("Initializing FlickdPipeline with debug mode")
            
        try:
            # Initialize components
            self.object_detector = ObjectDetector()
            self.product_matcher = ProductMatcher(catalog_path)
            self.vibe_classifier = VibeClassifier(vibes_list_path, debug=debug)
            
            if self.debug:
                logger.info("All pipeline components initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    def process_video(self, video_path: str, caption: str = None, hashtags: List[str] = None) -> Dict[str, Any]:
        """Process a video to detect objects, match products, and classify vibes."""
        if self.debug:
            logger.info(f"\nProcessing video: {video_path}")
            logger.info(f"Caption: {caption}")
            logger.info(f"Hashtags: {hashtags}")
            
        try:
            # Initialize results
            results = {
                "video_id": os.path.splitext(os.path.basename(video_path))[0],
                "vibes": [],
                "products": []
            }
            
            # Process video frames
            frames = self.object_detector.process_video(video_path)
            if not frames:
                logger.error("No frames extracted from video")
                return results
                
            if self.debug:
                logger.info(f"Processed {len(frames)} frames")
            
            # Detect objects
            objects = self.object_detector.detect_objects(frames)
            if self.debug:
                logger.info(f"Detected objects: {objects}")
            
            # Match products
            if objects:
                products = self.product_matcher.match_products(objects)
                if self.debug:
                    logger.info(f"Matched products: {products}")
                
                # Format products according to required output
                formatted_products = []
                for product in products:
                    formatted_product = {
                        "type": product["type"],
                        "color": product.get("color", "unknown"),
                        "match_type": product["match_type"].lower(),
                        "matched_product_id": product["id"],
                        "confidence": product["similarity"]
                    }
                    formatted_products.append(formatted_product)
                
                results["products"] = formatted_products
            
            # Classify vibes
            if caption or hashtags:
                if self.debug:
                    logger.info("Classifying vibes from caption and hashtags")
                vibes = self.vibe_classifier.process_caption(caption, hashtags)
                if self.debug:
                    logger.info(f"Detected vibes: {vibes}")
                
                # Format vibes according to required output
                results["vibes"] = [vibe["vibe"] for vibe in vibes]
            else:
                if self.debug:
                    logger.warning("No caption or hashtags provided for vibe classification")
                results["vibes"] = []
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                "video_id": os.path.splitext(os.path.basename(video_path))[0],
                "vibes": [],
                "products": [],
                "error": str(e)
            }
    
    def _format_video_id(self, filename: str) -> str:
        """Convert filename to reel_XXX format."""
        # Extract number from filename
        match = re.search(r'(\d+)', filename)
        if match:
            num = int(match.group(1))
            return f"reel_{num:03d}"
        return "reel_001"  # Default if no number found
    
    def _normalize_product_type(self, product_type: str) -> str:
        """Normalize product type to match expected format."""
        type_mapping = {
            'shirt': 'top',
            't-shirt': 'top',
            'blouse': 'top',
            'sweater': 'top',
            'hoodie': 'top',
            'pants': 'bottom',
            'jeans': 'bottom',
            'shorts': 'bottom',
            'skirt': 'bottom',
            'dress': 'dress',
            'jewelry': 'accessories',
            'bag': 'accessories',
            'shoes': 'accessories',
            'earrings': 'accessories',
            'necklace': 'accessories',
            'bracelet': 'accessories',
            'watch': 'accessories',
            'sunglasses': 'accessories',
            'hat': 'accessories',
            'scarf': 'accessories'
        }
        return type_mapping.get(product_type.lower(), product_type.lower())
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
