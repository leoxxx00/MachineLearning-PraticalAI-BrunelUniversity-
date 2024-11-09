import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# Load the dataset
file_path = '/Users/htet/Desktop/ML/assignments/data/data.csv'
data = pd.read_csv(file_path)
print("\nDataset loaded successfully.")

# Display basic information about the dataset
print("\nBasic Information:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

# Summary statistics of the original data
print("\nSummary Statistics of Original Data:")
original_summary = data.describe()
print(original_summary)

# Save summary statistics of original data
original_summary.to_csv("/Users/htet/Desktop/ML/assignments/data/original_summary.csv")

# Check for missing values and duplicates
print("\nMissing Values:")
print(data.isnull().sum())
duplicates = data.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")

# Save and visualize the distributions of numerical features before processing
num_columns = data.select_dtypes(include=[np.number]).columns.tolist()

print("\nPlotting distributions of numerical features before processing...")
for col in num_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Original Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/htet/Desktop/ML/assignments/plots/original_distribution_{col}.png")
    plt.show()

# Data Preprocessing - Centering, Standardization, Normalization, Scaling
print("\nStarting data preprocessing...")

# Centering Data
centered_data = data[num_columns] - data[num_columns].mean()
centered_summary = centered_data.describe()
print("\nCentered Data Summary Statistics:")
print(centered_summary)
centered_data.to_csv("/Users/htet/Desktop/ML/assignments/data/centered_data.csv")

# Standardization
standardized_data = pd.DataFrame(StandardScaler().fit_transform(data[num_columns]), columns=num_columns)
standardized_summary = standardized_data.describe()
print("\nStandardized Data Summary Statistics:")
print(standardized_summary)
standardized_data.to_csv("/Users/htet/Desktop/ML/assignments/data/standardized_data.csv")

# Normalization
normalized_data = pd.DataFrame(Normalizer().fit_transform(data[num_columns]), columns=num_columns)
normalized_summary = normalized_data.describe()
print("\nNormalized Data Summary Statistics:")
print(normalized_summary)
normalized_data.to_csv("/Users/htet/Desktop/ML/assignments/data/normalized_data.csv")

# Scaling
scaled_data = pd.DataFrame(MinMaxScaler().fit_transform(data[num_columns]), columns=num_columns)
scaled_summary = scaled_data.describe()
print("\nScaled Data Summary Statistics:")
print(scaled_summary)
scaled_data.to_csv("/Users/htet/Desktop/ML/assignments/data/scaled_data.csv")

# Combine all processed data for KNN application
combined_data = pd.concat([centered_data, standardized_data, normalized_data, scaled_data], axis=1)
combined_data.columns = [f"{col}_centered" for col in num_columns] + \
                        [f"{col}_standardized" for col in num_columns] + \
                        [f"{col}_normalized" for col in num_columns] + \
                        [f"{col}_scaled" for col in num_columns]
combined_data.to_csv("/Users/htet/Desktop/ML/assignments/data/combined_processed_data.csv", index=False)
print("\nCombined processed data saved for KNN model application.")

# Plotting the distributions of numerical features after each processing step

# Plot distributions after centering
print("\nPlotting distributions after centering...")
for col in num_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(centered_data[col], bins=30, kde=True, color='blue')
    plt.title(f'Centered Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/htet/Desktop/ML/assignments/plots/centered_distribution_{col}.png")
    plt.show()

# Plot distributions after standardization
print("\nPlotting distributions after standardization...")
for col in num_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(standardized_data[col], bins=30, kde=True, color='green')
    plt.title(f'Standardized Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/htet/Desktop/ML/assignments/plots/standardized_distribution_{col}.png")
    plt.show()

# Plot distributions after normalization
print("\nPlotting distributions after normalization...")
for col in num_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(normalized_data[col], bins=30, kde=True, color='purple')
    plt.title(f'Normalized Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/htet/Desktop/ML/assignments/plots/normalized_distribution_{col}.png")
    plt.show()

# Plot distributions after scaling
print("\nPlotting distributions after scaling...")
for col in num_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(scaled_data[col], bins=30, kde=True, color='red')
    plt.title(f'Scaled Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"/Users/htet/Desktop/ML/assignments/plots/scaled_distribution_{col}.png")
    plt.show()

# Comparison of summary statistics before and after transformations
print("\nComparing summary statistics before and after transformations:")
comparison = pd.concat({
    "Original": data[num_columns].describe().loc[['mean', 'std', 'min', 'max']],
    "Centered": centered_data.describe().loc[['mean', 'std', 'min', 'max']],
    "Standardized": standardized_data.describe().loc[['mean', 'std', 'min', 'max']],
    "Normalized": normalized_data.describe().loc[['mean', 'std', 'min', 'max']],
    "Scaled": scaled_data.describe().loc[['mean', 'std', 'min', 'max']]
}, axis=1)

print(comparison)
comparison.to_csv("/Users/htet/Desktop/ML/assignments/data/comparison_summary.csv")

# Plot and save the correlation matrix of numerical features
print("\nPlotting and saving the correlation matrix...")
correlation_matrix = data[num_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Numerical Features")
plt.savefig("/Users/htet/Desktop/ML/assignments/plots/correlation_matrix.png")
plt.show()

print("\nData preprocessing and plotting completed. All processed data and plots saved for future KNN model application.")
