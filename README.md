# Flickd AI Pipeline

A smart tagging and vibe classification engine for fashion videos, built for the Flickd AI Hackathon.

## Features

- Object detection using YOLOv8
- Product matching using CLIP and FAISS
- Vibe classification using DistilBERT
- RESTful API using FastAPI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required models:
```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('distilbert-base-uncased'); AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')"
```

3. Download YOLOv8 weights:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Project Structure

```
.
├── data/
│   ├── vibeslist.json    # List of supported vibes
│   ├── images.csv        # Product catalog
│   └── videos/           # Sample videos
├── src/
│   ├── object_detection.py
│   ├── product_matcher.py
│   ├── vibe_classifier.py
│   ├── pipeline.py
│   └── api.py
├── requirements.txt
└── README.md
```

## Usage

1. Start the API server:
```bash
python -m src.api
```

2. Send a POST request to `/process-video` with:
   - `video`: Video file
   - `caption`: Video caption (optional)
   - `hashtags`: JSON array of hashtags (optional)

Example using curl:
```bash
curl -X POST "http://localhost:8000/process-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@path/to/video.mp4" \
     -F "caption=Summer fashion look" \
     -F "hashtags=[\"fashion\", \"summer\", \"style\"]"
```

## Output Format

```json
{
    "video_id": "video_name",
    "vibes": ["Coquette", "Clean Girl"],
  "products": [
    {
            "type": "dress",
      "match_type": "similar",
            "matched_product_id": "prod_123",
      "confidence": 0.85
    }
  ]
}
```

## Notes

- The pipeline handles JSON serialization of numpy types automatically
- Videos are processed in memory and temporary files are cleaned up
- The API supports CORS for frontend integration
