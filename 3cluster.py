# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:04:36 2024
@author: eestllt
"""

# Import libraries
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  # Add this line to import matplotlib properly
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Check if libraries are imported correctly
print('Library versions:')
print('sklearn:', sklearn.__version__)
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('matplotlib:', matplotlib.__version__)

# Define paths
data_path = r'C:\Users\2361266\ML\data\data.csv'  # Input data
output_data_path = r'C:\Users\2361266\ML\data\data_kmeansconverted3.csv'  # Standardized output data
plots_dir = r'C:\Users\2361266\ML\plots'  # Output plots directory



# Import data
print('\nLoading data...')
df = pd.read_csv(data_path)
print('Data loaded successfully.')

# Display the beginning and the end of the dataset
print('\nFirst 5 rows of the dataset:')
print(df.head())
print('\nDataset information:')
df.info()

# Handle missing values
print('\nChecking for missing values...')
if df.isnull().values.any():
    print('Missing values found. Imputing missing values with column means...')
    df.fillna(df.mean(), inplace=True)
    print('Missing values imputed.')
else:
    print('No missing values found.')

# Select only numeric columns excluding any boolean columns
print('\nExtracting numeric features...')
X = df.select_dtypes(include=[np.number]).drop(columns=['ts']).values  # Drop 'ts' if it's not useful for clustering
print('Numeric features extracted:\n', X[:5])

# Plot original data (first two numeric features only for visualization)
plt.figure(figsize=(12, 5))
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', label='Original Data')
plt.title('Original Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig(f'{plots_dir}/original_data_plot_KNN3.png')
plt.show()

# Apply data centering before standardization
print('\nCentering data...')
X_centered = X - np.mean(X, axis=0)
print('Data after centering:\n', X_centered[:5])

# Plot centered data (first two features for visualization)
plt.figure(figsize=(12, 5))
plt.scatter(X_centered[:, 0], X_centered[:, 1], s=50, c='blue', label='Centered Data')
plt.title('Centered Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig(f'{plots_dir}/centered_data_plot_KNN3.png')
plt.show()

# Apply data standardization
print('\nApplying data standardization...')
std_X = StandardScaler().fit_transform(X_centered)
print('Data after standardization:\n', std_X[:5])

# Plot standardized data (first two features for visualization)
plt.figure(figsize=(12, 5))
plt.scatter(std_X[:, 0], std_X[:, 1], s=50, c='green', label='Standardized Data')
plt.title('Standardized Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig(f'{plots_dir}/standardized_data_plot_KNN3.png')
plt.show()

# Apply data scaling
print('\nApplying data scaling...')
scaled_X = MinMaxScaler().fit_transform(X_centered)
print('Data after scaling:\n', scaled_X[:5])

# Plot scaled data (first two features for visualization)
plt.figure(figsize=(12, 5))
plt.scatter(scaled_X[:, 0], scaled_X[:, 1], s=50, c='red', label='Scaled Data')
plt.title('Scaled Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig(f'{plots_dir}/scaled_data_plot_KNN3.png')
plt.show()

# Save the standardized data
print('\nSaving the standardized data...')
pd.DataFrame(std_X).to_csv(output_data_path, sep='\t', index=False)
print(f'Data saved successfully as {output_data_path}.')

# Build the KMeans model using standardized data with 3 clusters
print('\nBuilding the KMeans model...')
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10)
print('Fitting the model...')
y_kmeans = kmeans.fit_predict(std_X)
print('Model fitting completed.')
print('Predicted clusters:\n', y_kmeans)

# Plot 2D data (first two features for visualization)
print('\nVisualizing clusters in 2D...')
plt.figure(figsize=(10, 7))
plt.scatter(std_X[y_kmeans == 0, 0], std_X[y_kmeans == 0, 1], s=10, c='red', label='Cluster 1')
plt.scatter(std_X[y_kmeans == 1, 0], std_X[y_kmeans == 1, 1], s=10, c='blue', label='Cluster 2')
plt.scatter(std_X[y_kmeans == 2, 0], std_X[y_kmeans == 2, 1], s=10, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='purple', label='Centroids')
plt.title('Clusters (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig(f'{plots_dir}/clusters_2d_plot_KNN3.png')
plt.show()
print('2D visualization completed.')
