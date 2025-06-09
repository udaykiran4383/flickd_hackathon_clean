import pandas as pd
import json

# Read the Excel file
df = pd.read_excel('data/product_data.xlsx')

# Create a new DataFrame with the required columns
catalog_df = pd.DataFrame()

# Copy essential columns
catalog_df['id'] = df['id']
catalog_df['title'] = df['title']
catalog_df['product_type'] = df['product_type']
catalog_df['product_tags'] = df['product_tags']
catalog_df['product_collections'] = df['product_collections']

# Save to CSV
catalog_df.to_csv('data/catalog.csv', index=False)

print("Catalog created successfully!")
print("\nFirst few rows of the catalog:")
print(catalog_df.head())
print("\nTotal products:", len(catalog_df)) 