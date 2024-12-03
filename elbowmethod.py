# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:04:36 2024
@author: eestllt
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Define paths
file_path =  '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/data.csv'
plot_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/plots' 

# Create plot directory if it doesn't exist
os.makedirs(plot_path, exist_ok=True)

# Load the dataset
data = pd.read_csv(file_path)
print("\nDataset loaded successfully.")

# Display basic information
print("\nBasic Information:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

# Identify numerical columns
num_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Preprocess - Standardization
print("\nStarting data preprocessing...")
standardized_data = pd.DataFrame(StandardScaler().fit_transform(data[num_columns]), columns=num_columns)

# Select the standardized data for KMeans clustering
X = standardized_data.values

# Apply PCA if data is high-dimensional
from sklearn.decomposition import PCA
if X.shape[1] > 10:
    print('\nApplying PCA to reduce dimensions...')
    pca = PCA(n_components=10)
    X = pca.fit_transform(X)
    print('Data after PCA:\n', X[:5])

# Elbow Method to find optimal clusters
wcss = []
print("\nCalculating WCSS for different numbers of clusters...")
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(f'Number of Clusters: {i}, WCSS: {kmeans.inertia_}')

# Plot WCSS
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), wcss, color='skyblue')
plt.title('Elbow Method - WCSS by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig(os.path.join(plot_path, 'wcss_bar_chart.png'))
plt.show()

print('\nElbow method completed successfully.')