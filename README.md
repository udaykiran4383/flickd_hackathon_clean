# Flickd AI Hackathon

A video-based fashion item detection and vibe classification system.

## Features

- Video frame extraction
- Fashion item detection using DeepFashion2 YOLOv8
- Product matching using CLIP embeddings
- Vibe classification using rule-based NLP
- JSON output generation

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the DeepFashion2 model:
```bash
wget https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt
```

4. Pre-compute product embeddings:
```bash
python scripts/precompute_embeddings.py
```

## Usage

1. Place your videos in the `videos/` directory
2. Run the test script:
```bash
python test_multiple_videos.py
```

3. Check the results in `test_results/`

## Project Structure

```
.
├── data/
│   ├── catalog.csv
│   ├── vibes_list.json
│   └── product_embeddings.json
├── src/
│   ├── object_detector.py
│   ├── clip_processor.py
│   ├── vibe_classifier.py
│   └── product_matcher.py
├── scripts/
│   └── precompute_embeddings.py
├── test_videos/
│   └── *.mp4
├── test_results/
│   └── *.json
├── requirements.txt
└── README.md
```

## Models Used

- DeepFashion2 YOLOv8 (mAP₅₀ = 0.849)
- CLIP ViT-B/32 for image embeddings
- FAISS for fast similarity search

## Output Format

```json
{
  "video_id": "reel_001",
  "vibes": ["Coquette", "Clean Girl"],
  "products": [
    {
      "type": "top",
      "color": "white",
      "matched_product_id": "prod_002",
      "match_type": "exact",
      "confidence": 0.93
    }
  ]
}
```

## Notes

- The pipeline handles JSON serialization of numpy types automatically
- Videos are processed in memory and temporary files are cleaned up
- The API supports CORS for frontend integration
