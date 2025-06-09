import pandas as pd
import requests
from tqdm import tqdm
import csv

# Load data
catalog = pd.read_csv('data/catalog.csv')
images = pd.read_csv('data/images.csv')

# Get first image per product id
first_images = images.groupby('id')['image_url'].first().reset_index()

# Merge into catalog
catalog = catalog.merge(first_images, how='left', left_on='id', right_on='id')

# Check if image URL is valid (HTTP 200)
def check_url(url):
    try:
        r = requests.head(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False

tqdm.pandas()
catalog['image_url'] = catalog['image_url'].progress_apply(lambda x: x if pd.isna(x) or check_url(x) else '')

# Save updated catalog
catalog.to_csv('data/catalog.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

# Print summary
num_with_image = catalog['image_url'].astype(bool).sum()
print(f"image_url column present: {'image_url' in catalog.columns}")
print(f"Products with valid image: {num_with_image} of {len(catalog)}") 