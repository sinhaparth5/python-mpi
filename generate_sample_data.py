import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import os

# Create data directory if it does't exits
os.makedirs('data', exist_ok=True)

# generate synthetic data 
n_samples = 1000
n_features = 2
n_clusters = 3

# generate
X, y  = make_blobs(n_samples=n_samples,
                   n_features=n_features,
                   centers=n_clusters,
                   cluster_std=1.0,
                   random_state=42)

df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
df['label'] = y

output_path = 'data/sample_data.csv'
df.to_csv(output_path, index=False)

print(f"Generated sample dataset with {n_samples} samples and {n_features} features")
print(f"Saved to: {output_path}")

# Display first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Display basic statistics
print("\nBasic statistics:")
print(df.describe())