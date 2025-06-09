import os
import logging
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
from pathlib import Path

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_path: str = None, debug: bool = False):
        """Initialize the object detector with YOLOv8 models."""
        self.debug = debug
        try:
            # Load DeepFashion2 model
            if model_path and os.path.exists(model_path):
                self.fashion_model = YOLO(model_path)
                logger.info(f"Loaded DeepFashion2 model from {model_path}")
            else:
                # Fallback to default YOLOv8 model
                self.fashion_model = YOLO('yolov8n.pt')
                logger.warning("Using default YOLOv8 model as DeepFashion2 model not found")
            
            # Fashion class mapping
            self.fashion_classes = {
                0: "short_sleeved_shirt",
                1: "long_sleeved_shirt",
                2: "short_sleeved_outwear",
                3: "long_sleeved_outwear",
                4: "vest",
                5: "sling",
                6: "shorts",
                7: "trousers",
                8: "skirt",
                9: "short_sleeved_dress",
                10: "long_sleeved_dress",
                11: "vest_dress",
                12: "sling_dress"
            }
            
            if self.debug:
                logger.info("ObjectDetector initialized with debug mode")
                
        except Exception as e:
            logger.error(f"Error initializing ObjectDetector: {str(e)}")
            raise

    def process_video(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video."""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames (2 frames per second for better coverage)
            sample_interval = int(fps / 2)
            
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_number % sample_interval == 0:
                    # Enhance frame quality
                    frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
                    frames.append(frame)
                    
                frame_number += 1
                
            cap.release()
            
            if self.debug:
                logger.info(f"Extracted {len(frames)} frames from video")
                
            return frames
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return []

    def detect_objects(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect objects in frames using YOLOv8."""
        try:
            detected_objects = []
            
            for frame_idx, frame in enumerate(frames):
                # Run fashion detection with lower confidence threshold
                fashion_results = self.fashion_model(frame, conf=0.15)[0]  # Lowered from 0.25 to 0.15
                
                # Process fashion detections
                for box in fashion_results.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id in self.fashion_classes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Add detection
                        detected_objects.append({
                            'type': self.fashion_classes[class_id],
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'frame_number': frame_idx,
                            'frame': frame
                        })
                        
                        if self.debug:
                            logger.info(f"Detected {self.fashion_classes[class_id]} with confidence {confidence:.2f}")
            
            if self.debug:
                logger.info(f"Detected {len(detected_objects)} objects across {len(frames)} frames")
                
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return [] 