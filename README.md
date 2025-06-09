# Flickd - Fashion Video Analysis Pipeline

A powerful pipeline for analyzing fashion videos to detect clothing items, match them with products, and classify aesthetic vibes.

## Project Journey & Insights

### Initial Challenges
- Started with basic object detection using YOLO, but needed more specialized fashion detection
- Found DeepFashion2 dataset and model which was better suited for clothing detection
- Realized we needed both visual and semantic understanding for accurate product matching
- Discovered CLIP model's effectiveness for fashion item matching

### Key Learnings
1. **Vibe Classification**
   - Initially used only text-based classification (captions/hashtags)
   - Realized products themselves carry strong aesthetic signals
   - Combined both approaches for more accurate vibe detection
   - Created a comprehensive vibe dictionary with product attributes

2. **Product Matching**
   - Started with simple color/type matching
   - Evolved to use CLIP embeddings for semantic understanding
   - Added confidence thresholds to ensure quality matches
   - Implemented duplicate detection to avoid redundant suggestions

3. **Performance Optimization**
   - Cached CLIP embeddings for faster processing
   - Implemented batch processing for video frames
   - Added confidence thresholds to reduce false positives

## Project Structure

```
flickd/
├── src/                    # Core functionality
│   ├── pipeline.py         # Main pipeline implementation
│   ├── vibe_classifier.py  # Vibe classification logic
│   ├── product_matcher.py  # Product matching logic
│   ├── object_detector.py  # Object detection logic
│   ├── clip_processor.py   # CLIP model processing
│   └── __init__.py
├── data/                   # Data directory
│   ├── videos/            # Input videos
│   └── catalog/           # Product catalog
├── outputs/               # Generated results
├── tests/                 # Test files
└── requirements.txt       # Project dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flickd.git
cd flickd
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
# DeepFashion2 YOLO model
wget https://github.com/yourusername/flickd/releases/download/v1.0/deepfashion2_yolov8s-seg.pt
```

### Configuration

1. Prepare your product catalog:
   - Place product images in `data/catalog/`
   - Ensure images are clear and well-lit
   - Include product metadata (type, color, style)

2. Configure API settings (if using API):
   - Update `API_URL` in `test_api.py`
   - Set appropriate rate limits

## Usage

### Basic Usage

```python
from src.pipeline import FlickdPipeline

# Initialize pipeline
pipeline = FlickdPipeline(
    vibes_list_path="data/vibeslist.json",
    catalog_path="data/catalog",
    debug=True
)

# Process a video
results = pipeline.process_video(
    video_path="data/videos/example.mp4",
    caption="Summer vibes with my new outfit!",
    hashtags=["#fashion", "#summer"]
)

print(results)
```

### API Usage

```bash
# Start the API server
python -m src.api.server

# Test the API
python test_api.py
```

## Common Mistakes & Solutions

1. **Model Loading Issues**
   - Problem: CUDA out of memory
   - Solution: Reduce batch size in `object_detector.py`
   - Problem: Model not found
   - Solution: Check model path and download if missing

2. **Product Matching Issues**
   - Problem: Low confidence matches
   - Solution: Adjust thresholds in `product_matcher.py`
   - Problem: Duplicate products
   - Solution: Check `max_duplicate_distance` setting

3. **Vibe Classification Issues**
   - Problem: Inconsistent vibe detection
   - Solution: Update vibe keywords in `vibe_classifier.py`
   - Problem: Missing vibes
   - Solution: Check product attributes in vibe definitions

## Performance Tips

1. **Video Processing**
   - Use shorter videos (15-30 seconds)
   - Ensure good lighting and clear shots
   - Avoid rapid movements

2. **Product Catalog**
   - Use high-quality product images
   - Include multiple angles
   - Maintain consistent background

3. **System Optimization**
   - Use GPU acceleration
   - Enable caching for embeddings
   - Monitor memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepFashion2 dataset and model
- CLIP model by OpenAI
- YOLOv8 by Ultralytics
