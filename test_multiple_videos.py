import os
import logging
import json
from pathlib import Path
from src.object_detector import ObjectDetector
from src.clip_processor import FashionCLIPProcessor
from src.vibe_classifier import VibeClassifier
from src.product_matcher import ProductMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(video_path, object_detector, clip_processor, vibe_classifier, product_matcher):
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

        # Get vibes from caption (using a placeholder for now)
        caption = "✨ Comment 'LINK' and I'll DM you the details! 🤍 GRWM in this easy-breezy cotton vest + skirt set — made in linen, made for summer! 🌞 #GRWM #LinenSet #SummerOutfit #CoOrdSet #OOTDIndia #GRWMReel"
        vibes = vibe_classifier.process_caption(caption)
        vibe_names = [v['vibe'] for v in vibes]

        # Process each detected object
        products = []
        for obj in detected_objects:
            if obj['confidence'] < 0.25:  # Lower threshold to match detector
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
                    product = {
                        "type": obj['type'],
                        "color": match.get('color', 'unknown'),
                        "matched_product_id": match.get('id', 'unknown'),
                        "match_type": "exact" if match.get('confidence', 0) > 0.9 else "similar",
                        "confidence": float(match.get('confidence', 0))
                    }
                    products.append(product)

        # Create result in required format
        result = {
            "video_id": video_id,
            "vibes": vibe_names,
            "products": products
        }

        return result

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize components
    vibe_classifier = VibeClassifier('data/vibeslist.json', debug=True)
    object_detector = ObjectDetector('deepfashion2_yolov8s-seg.pt', debug=True)
    clip_processor = FashionCLIPProcessor(debug=True)
    product_matcher = ProductMatcher('data/product_embeddings.npy', debug=True)

    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # List of videos to process
    video_paths = [
        'data/videos/2025-05-22_08-25-12_UTC.mp4',
        'data/videos/2025-05-27_13-46-16_UTC.mp4',
        'data/videos/2025-05-28_13-40-09_UTC.mp4'
    ]

    # Process each video
    all_results = []
    for video_path in video_paths:
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        logger.info(f"Processing {video_path}")
        result = process_video(
            video_path,
            object_detector,
            clip_processor,
            vibe_classifier,
            product_matcher
        )

        if result:
            # Save individual result
            output_path = os.path.join(output_dir, f"{Path(video_path).stem}_results.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {output_path}")
            all_results.append(result)

    # Save combined results
    if all_results:
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Combined results saved to {combined_path}")
    else:
        logger.warning("No results to save")

if __name__ == "__main__":
    main()
