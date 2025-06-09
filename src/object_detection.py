import os
import cv2
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
import torch
from ultralytics import YOLO
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector with improved model and logging."""
        try:
            # Initialize logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
            # Load YOLOv8x model for better accuracy
            self.model = YOLO('yolov8x.pt')
            self.logger.info("YOLOv8x model loaded successfully")
            
            # Define fashion-related classes
            self.fashion_classes = {
                0: 'person',  # For full body detection
                15: 'handbag',
                16: 'backpack',
                27: 'handbag',
                28: 'umbrella',
                31: 'handbag',
                32: 'backpack',
                33: 'umbrella',
                44: 'bottle',  # For water bottles, etc.
                45: 'wine glass',  # For accessories
                46: 'cup',  # For accessories
                47: 'fork',  # For accessories
                48: 'knife',  # For accessories
                49: 'spoon',  # For accessories
                50: 'bowl',  # For accessories
                51: 'banana',  # For accessories
                52: 'apple',  # For accessories
                53: 'sandwich',  # For accessories
                54: 'orange',  # For accessories
                55: 'broccoli',  # For accessories
                56: 'carrot',  # For accessories
                57: 'hot dog',  # For accessories
                58: 'pizza',  # For accessories
                59: 'donut',  # For accessories
                60: 'cake',  # For accessories
                61: 'chair',  # For accessories
                62: 'couch',  # For accessories
                63: 'potted plant',  # For accessories
                64: 'bed',  # For accessories
                65: 'dining table',  # For accessories
                66: 'toilet',  # For accessories
                67: 'tv',  # For accessories
                68: 'laptop',  # For accessories
                69: 'mouse',  # For accessories
                70: 'remote',  # For accessories
                71: 'keyboard',  # For accessories
                72: 'cell phone',  # For accessories
                73: 'microwave',  # For accessories
                74: 'oven',  # For accessories
                75: 'toaster',  # For accessories
                76: 'sink',  # For accessories
                77: 'refrigerator',  # For accessories
                78: 'book',  # For accessories
                79: 'clock',  # For accessories
                80: 'vase',  # For accessories
                81: 'scissors',  # For accessories
                82: 'teddy bear',  # For accessories
                83: 'hair drier',  # For accessories
                84: 'toothbrush'  # For accessories
            }
            
            # Initialize previous detections
            self.previous_detections = None
            
        except Exception as e:
            self.logger.error(f"Error initializing ObjectDetector: {e}")
            raise
        
    def extract_frames(self, video_path: str, frame_interval: int = 5) -> List[np.ndarray]:  # More frequent sampling
        """Extract frames from video at specified intervals."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Enhance frame quality
                frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
                frames.append(frame)
            frame_count += 1
            
        cap.release()
        return frames
    
    def post_process_bbox(self, x1: float, y1: float, x2: float, y2: float, frame_shape: tuple) -> tuple:
        """Post-process bounding box coordinates for better accuracy."""
        # Ensure coordinates are within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_shape[1], x2)
        y2 = min(frame_shape[0], y2)
        
        # Ensure minimum size
        min_size = 50
        if x2 - x1 < min_size:
            center_x = (x1 + x2) / 2
            x1 = center_x - min_size / 2
            x2 = center_x + min_size / 2
        if y2 - y1 < min_size:
            center_y = (y1 + y2) / 2
            y1 = center_y - min_size / 2
            y2 = center_y + min_size / 2
            
        # Ensure aspect ratio is reasonable
        aspect_ratio = (x2 - x1) / (y2 - y1)
        if aspect_ratio > 2:  # Too wide
            center_x = (x1 + x2) / 2
            width = (y2 - y1) * 2
            x1 = center_x - width / 2
            x2 = center_x + width / 2
        elif aspect_ratio < 0.5:  # Too tall
            center_y = (y1 + y2) / 2
            height = (x2 - x1) * 2
            y1 = center_y - height / 2
            y2 = center_y + height / 2
            
        return x1, y1, x2, y2
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in a frame with improved accuracy and post-processing."""
        try:
            # Enhance frame quality
            frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
            
            # Run YOLOv8 inference
            results = self.model(frame, conf=0.3)  # Lower confidence threshold for more detections
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only process fashion-related classes
                    if class_id in self.fashion_classes:
                        # Post-process bounding box
                        x1, y1, x2, y2 = self.post_process_bbox(x1, y1, x2, y2, frame.shape)
                        
                        # Get class name
                        class_name = self.fashion_classes[class_id]
                        
                        # Add detection
                        detections.append({
                            'frame': frame,
                            'frame_number': 0,  # Will be updated by pipeline
                            'bbox': [x1, y1, x2, y2],
                            'class_name': class_name,
                            'confidence': confidence
                        })
            
            # Apply temporal consistency if we have previous detections
            if self.previous_detections is not None:
                detections = self.apply_temporal_consistency(detections, self.previous_detections)
            
            # Update previous detections
            self.previous_detections = detections
                
            return detections
            
        except Exception as e:
            self.logger.error(f"Error detecting objects: {e}")
            return []
    
    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Process entire video and return all detections with temporal consistency."""
        frames = self.extract_frames(video_path)
        all_detections = []
        
        for frame_idx, frame in enumerate(frames):
            detections = self.detect_objects(frame)
            
            # Update frame number
            for detection in detections:
                detection["frame_number"] = frame_idx
            
            all_detections.extend(detections)
            
        return all_detections
    
    def apply_temporal_consistency(self, current_detections: List[Dict], prev_detections: List[Dict]) -> List[Dict]:
        """Apply temporal consistency to current detections using previous frame."""
        if not prev_detections:
            return current_detections
            
        consistent_detections = []
        for curr_det in current_detections:
            best_match = None
            best_iou = 0.0
            
            for prev_det in prev_detections:
                iou = self.calculate_iou(curr_det['bbox'], prev_det['bbox'])
                if iou > best_iou and iou > 0.5:  # Only match if IOU > 0.5
                    best_iou = iou
                    best_match = prev_det
            
            if best_match:
                # Average bounding boxes
                curr_det['bbox'] = self.average_bbox(curr_det['bbox'], best_match['bbox'])
                # Use higher confidence
                curr_det['confidence'] = max(curr_det['confidence'], best_match['confidence'])
            
            consistent_detections.append(curr_det)
            
        return consistent_detections
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def average_bbox(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """Average two bounding boxes."""
        return [
            (bbox1[0] + bbox2[0]) / 2,
            (bbox1[1] + bbox2[1]) / 2,
            (bbox1[2] + bbox2[2]) / 2,
            (bbox1[3] + bbox2[3]) / 2
        ]

    def _get_dominant_color(
        self,
        frame: np.ndarray,
        bbox: tuple
    ) -> str:
        """
        Get the dominant color in a bounding box.
        
        Args:
            frame: Image frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Dominant color name
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2]
            
            # Convert to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Reshape to 2D array of pixels
            pixels = roi.reshape(-1, 3)
            
            # Get unique colors and their counts
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Get the most common color
            dominant_color = unique_colors[counts.argmax()]
            
            # Map RGB to color name
            return self._rgb_to_color_name(dominant_color)
            
        except Exception as e:
            logger.error(f"Error getting dominant color: {str(e)}")
            return "unknown"
            
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """
        Convert RGB values to color name.
        
        Args:
            rgb: RGB values as numpy array
            
        Returns:
            Color name string
        """
        # Define basic colors and their RGB values
        colors = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "purple": [128, 0, 128],
            "orange": [255, 165, 0],
            "pink": [255, 192, 203],
            "brown": [165, 42, 42],
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "gray": [128, 128, 128]
        }
        
        # Calculate distances to each color
        distances = {}
        for name, value in colors.items():
            distances[name] = np.sqrt(np.sum((rgb - value) ** 2))
        
        # Return the closest color
        return min(distances.items(), key=lambda x: x[1])[0] 