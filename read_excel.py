import pandas as pd

# Read the Excel file
df = pd.read_excel('data/product_data.xlsx')

# Display basic information about the DataFrame
print("\nDataFrame Info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nSample values for each column:")
for col in df.columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head()) 