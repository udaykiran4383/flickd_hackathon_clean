import os
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from src.object_detector import ObjectDetector
from src.clip_processor import FashionCLIPProcessor
from src.vibe_classifier import VibeClassifier
from src.product_matcher import ProductMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    object_detection_confidence: float = 0.25  # YOLO detection confidence threshold
    min_distance_threshold: float = 0.7    # Maximum distance for a match
    max_distance_threshold: float = 0.45   # Minimum distance for a match
    exact_match_threshold: float = 0.9     # Distance threshold for exact matches (>0.9 similarity)
    similar_match_threshold: float = 0.75  # Distance threshold for similar matches (0.75-0.9 similarity)
    max_products_per_type: int = 2         # Limit products per type to get 2-4 total
    max_vibes: int = 3                     # Maximum number of vibes to return
    min_confidence: float = 0.75           # Minimum confidence for product matches
    max_duplicate_distance: float = 0.05   # Maximum distance between duplicates

def distance_to_confidence(distance: float, config: ProcessingConfig) -> float:
    """Convert distance to confidence score."""
    if distance > config.min_distance_threshold:
        return 0.0
    if distance < config.max_distance_threshold:
        return 1.0
    # Linear interpolation between thresholds
    normalized = (distance - config.max_distance_threshold) / (config.min_distance_threshold - config.max_distance_threshold)
    return 1.0 - normalized

def extract_caption(video_path: str) -> str:
    """Extract caption from either .txt or .json file."""
    caption_path = video_path.replace('.mp4', '.txt')
    json_path = video_path.replace('.mp4', '.json')
    caption = ""

    # Try .txt first
    if os.path.exists(caption_path):
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        except Exception as e:
            logger.warning(f"Error reading caption from TXT for {video_path}: {e}")

    # If .txt is missing or empty, try .json
    if not caption and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle different JSON structures
                if 'node' in data:
                    edges = data.get('node', {}).get('edge_media_to_caption', {}).get('edges', [])
                    if edges and 'node' in edges[0] and 'text' in edges[0]['node']:
                        caption = edges[0]['node']['text'].strip()
                elif 'caption' in data:
                    caption = data['caption'].strip()
        except Exception as e:
            logger.warning(f"Error reading caption from JSON for {video_path}: {e}")

    return caption

def deduplicate_products(products: List[Dict[str, Any]], config: ProcessingConfig) -> List[Dict[str, Any]]:
    """Remove duplicate products based on type, color, and distance."""
    if not products:
        return []
        
    # Sort by confidence in descending order
    products.sort(key=lambda x: x['confidence'], reverse=True)
    
    unique_products = []
    seen_types = set()
    
    for product in products:
        product_type = product['type']
        if product_type not in seen_types:
            seen_types.add(product_type)
            unique_products.append(product)
            continue
            
        # Check if this is a duplicate of an existing product
        is_duplicate = False
        for existing in unique_products:
            if (existing['type'] == product_type and 
                existing['color'] == product['color'] and
                abs(existing['confidence'] - product['confidence']) < config.max_duplicate_distance):
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_products.append(product)
    
    return unique_products

def validate_result(result: Dict[str, Any]) -> bool:
    """Validate the result structure and content."""
    required_fields = ['video_id', 'vibes', 'products']
    if not all(field in result for field in required_fields):
        return False
    
    if not isinstance(result['vibes'], list) or not isinstance(result['products'], list):
        return False
    
    if not result['video_id']:
        return False
    
    # Validate product structure
    for product in result['products']:
        required_product_fields = ['type', 'color', 'matched_product_id', 'match_type', 'confidence']
        if not all(field in product for field in required_product_fields):
            return False
        if not isinstance(product['confidence'], (int, float)):
            return False
        if product['confidence'] < 0 or product['confidence'] > 1:
            return False
    
    return True

def process_video(
    video_path: str,
    object_detector: ObjectDetector,
    clip_processor: FashionCLIPProcessor,
    vibe_classifier: VibeClassifier,
    product_matcher: ProductMatcher,
    config: ProcessingConfig
) -> Optional[Dict[str, Any]]:
    """Process a single video and return results."""
    try:
        # Extract frames
        frames = object_detector.process_video(video_path)
        if not frames:
            logger.warning(f"No frames extracted from {video_path}")
            return None

        # Detect objects
        detected_objects = object_detector.detect_objects(frames)
        if not detected_objects:
            logger.warning(f"No objects detected in {video_path}")
            return None

        # Get video ID from filename
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Extract caption
        caption = extract_caption(video_path)
        if not caption:
            logger.warning(f"No caption found for {video_path}")

        # Get vibes from caption
        vibes = vibe_classifier.process_caption(caption)
        vibe_names = [v['vibe'] for v in vibes[:config.max_vibes]]

        # Process each detected object
        products = []
        for obj in detected_objects:
            if obj['confidence'] < config.object_detection_confidence:
                continue
                
            # Get CLIP embedding for the object
            frame = frames[obj['frame_number']]
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
                
            embedding = clip_processor.get_embedding(cropped)
            
            # Find matching products
            matches = product_matcher.find_matches(embedding, obj['type'])
            if matches:
                for match in matches:
                    distance = float(match.get('distance', 1.0))
                    confidence = distance_to_confidence(distance, config)
                    
                    if confidence < config.min_confidence:
                        continue
                        
                    product = {
                        "type": obj['type'],
                        "color": match.get('color', 'unknown'),
                        "matched_product_id": match.get('matched_product_id', 'unknown'),
                        "match_type": "exact" if distance < config.exact_match_threshold else "similar",
                        "confidence": confidence
                    }
                    products.append(product)
                    logger.debug(f"Added product: {product} with distance {distance} and confidence {confidence}")

        # Deduplicate products
        products = deduplicate_products(products, config)

        # Create result in required format
        result = {
            "video_id": video_id,
            "vibes": vibe_names,
            "products": products
        }

        # Validate result
        if not validate_result(result):
            logger.error(f"Invalid result structure for {video_path}")
            return None

        # Log processing summary
        logger.info(f"Processed {video_id}:")
        logger.info(f"  - Detected {len(detected_objects)} objects")
        logger.info(f"  - Found {len(products)} unique products")
        if products:
            logger.info(f"  - Product types: {', '.join(set(p['type'] for p in products))}")
            logger.info(f"  - Average confidence: {sum(p['confidence'] for p in products)/len(products):.2f}")
        logger.info(f"  - Identified {len(vibe_names)} vibes: {', '.join(vibe_names)}")

        return result

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}", exc_info=True)
        return None

def main():
    # Initialize configuration
    config = ProcessingConfig()

    # Initialize components
    vibe_classifier = VibeClassifier('data/vibeslist.json', debug=True)
    object_detector = ObjectDetector('deepfashion2_yolov8s-seg.pt', debug=True)
    clip_processor = FashionCLIPProcessor(debug=True)
    product_matcher = ProductMatcher('data/product_embeddings.npy', debug=True)

    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Get all video files
    video_dir = "data/videos"
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                  if f.endswith('.mp4')]

    total_videos = len(video_paths)
    logger.info(f"Found {total_videos} videos to process")

    # Process each video
    all_results = []
    successful_videos = 0
    total_objects = 0
    total_products = 0
    
    for idx, video_path in enumerate(video_paths, 1):
        logger.info(f"Processing video {idx}/{total_videos}: {video_path}")
        result = process_video(
            video_path,
            object_detector,
            clip_processor,
            vibe_classifier,
            product_matcher,
            config
        )

        if result:
            # Update statistics
            successful_videos += 1
            total_objects += len(result.get('products', []))
            total_products += len(result.get('products', []))
            
            # Save individual result
            output_path = os.path.join(output_dir, f"{Path(video_path).stem}_results.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {output_path}")
            all_results.append(result)
        else:
            logger.warning(f"No results generated for {video_path}")

    # Save combined results
    if all_results:
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Log final statistics
        logger.info("\nProcessing Summary:")
        logger.info(f"  - Successfully processed {successful_videos}/{total_videos} videos")
        logger.info(f"  - Total objects detected: {total_objects}")
        logger.info(f"  - Total unique products found: {total_products}")
        logger.info(f"  - Average products per video: {total_products/successful_videos:.2f}")
        logger.info(f"Combined results saved to {combined_path}")
    else:
        logger.warning("No results to save")

if __name__ == "__main__":
    main()
