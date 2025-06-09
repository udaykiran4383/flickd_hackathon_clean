import pandas as pd
import json
import os
from pathlib import Path

# Define paths
VIDEOS_DIR = "/Users/uday/flick hackthon/data/videos"
IMAGES_CSV = "/Users/uday/flick hackthon/data/images.csv"
PRODUCT_EXCEL = "/Users/uday/flick hackthon/data/product_data.xlsx"
VIBES_JSON = "/Users/uday/flick hackthon/data/vibeslist.json"
OUTPUT_DIR = "outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load product catalog from both sources
try:
    images_df = pd.read_csv(IMAGES_CSV)
    print("Loaded images.csv successfully")
except Exception as e:
    print(f"Error loading images.csv: {e}")
    images_df = None

try:
    product_df = pd.read_excel(PRODUCT_EXCEL)
    print("Loaded product_data.xlsx successfully")
except Exception as e:
    print(f"Error loading product_data.xlsx: {e}")
    product_df = None

# Load vibes list
try:
    with open(VIBES_JSON, 'r') as f:
        vibes_list = json.load(f)
    print("Loaded vibes list successfully")
except Exception as e:
    print(f"Error loading vibes list: {e}")
    vibes_list = []

# Get list of video files
video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]
print("\nVideo files found:")
for video in video_files:
    print(f"- {video}")

# Print data summaries
if images_df is not None:
    print(f"\nImages catalog shape: {images_df.shape}")
    print("Columns:", images_df.columns.tolist())

if product_df is not None:
    print(f"\nProduct data shape: {product_df.shape}")
    print("Columns:", product_df.columns.tolist())

print(f"\nNumber of vibes: {len(vibes_list)}")
print("Vibes:", vibes_list) 