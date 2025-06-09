import pandas as pd
import requests
from urllib.parse import urlparse, parse_qs

def check_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except:
        return False

# Read the Excel file
df = pd.read_excel('data/product_data.xlsx')

# Check if there's an image URL column
print("Columns in Excel file:", df.columns.tolist())

# If there's an image URL column, check a few URLs
if 'image_url' in df.columns:
    print("\nChecking first 5 URLs from Excel file:")
    for url in df['image_url'].head():
        print(f"URL: {url}")
        print(f"Status: {'Valid' if check_url(url) else 'Invalid'}\n") 