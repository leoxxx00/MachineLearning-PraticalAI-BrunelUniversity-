import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# File paths for balanced data and preprocessed data output
balanced_true_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/balanced_true_data.csv'
balanced_false_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/balanced_false_data.csv'
preprocessed_true_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/preprocessed_true_data.csv'
preprocessed_false_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/preprocessed_false_data.csv'

# File path for saving the KNN model
knn_model_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/knn_model.pkl'

# Load the balanced 'motion = True' and 'motion = False' data
print(f"Loading balanced 'motion = True' data from {balanced_true_file_path}...")
balanced_true_data = pd.read_csv(balanced_true_file_path)

print(f"Loading balanced 'motion = False' data from {balanced_false_file_path}...")
balanced_false_data = pd.read_csv(balanced_false_file_path)

# Check the shape of the loaded data
print(f"\nLoaded 'motion = True' data: {balanced_true_data.shape}")
print(f"Loaded 'motion = False' data: {balanced_false_data.shape}")

# Separate features and target variable ('motion')
# Assuming 'motion' column contains the target variable
true_data_features = balanced_true_data.drop(columns=['motion'])
false_data_features = balanced_false_data.drop(columns=['motion'])

# Normalize and Standardize the features

# Initialize the scalers for different methods
scaler_standard = StandardScaler()  # Standardization (mean=0, std=1)
scaler_minmax = MinMaxScaler()      # Min-Max normalization (range [0, 1])

# Normalize and standardize the data
print("\nNormalizing and standardizing 'motion = True' data...")
true_data_standardized = scaler_standard.fit_transform(true_data_features)
true_data_normalized = scaler_minmax.fit_transform(true_data_features)

print("\nNormalizing and standardizing 'motion = False' data...")
false_data_standardized = scaler_standard.fit_transform(false_data_features)
false_data_normalized = scaler_minmax.fit_transform(false_data_features)

# Convert normalized and standardized data back to DataFrame
true_data_standardized_df = pd.DataFrame(true_data_standardized, columns=true_data_features.columns)
false_data_standardized_df = pd.DataFrame(false_data_standardized, columns=false_data_features.columns)

true_data_normalized_df = pd.DataFrame(true_data_normalized, columns=true_data_features.columns)
false_data_normalized_df = pd.DataFrame(false_data_normalized, columns=false_data_features.columns)

# Add the target variable back
true_data_standardized_df['motion'] = balanced_true_data['motion']
false_data_standardized_df['motion'] = balanced_false_data['motion']

true_data_normalized_df['motion'] = balanced_true_data['motion']
false_data_normalized_df['motion'] = balanced_false_data['motion']

# Save the preprocessed data (standardized and normalized versions)
print(f"\nSaving preprocessed 'motion = True' data (standardized) to {preprocessed_true_file_path}...")
true_data_standardized_df.to_csv(preprocessed_true_file_path, index=False)
print(f"Preprocessed 'motion = True' data (standardized) saved successfully.")

print(f"\nSaving preprocessed 'motion = False' data (standardized) to {preprocessed_false_file_path}...")
false_data_standardized_df.to_csv(preprocessed_false_file_path, index=False)
print(f"Preprocessed 'motion = False' data (standardized) saved successfully.")

# Prepare the data for training a K-Nearest Neighbors (KNN) model

# Combine the datasets (true + false) for model training
X = pd.concat([true_data_standardized_df.drop(columns=['motion']),
               false_data_standardized_df.drop(columns=['motion'])], axis=0)
y = pd.concat([true_data_standardized_df['motion'], false_data_standardized_df['motion']], axis=0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN model (you can adjust n_neighbors as needed)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nKNN Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained KNN model for later use
print(f"\nSaving the KNN model to {knn_model_path}...")
joblib.dump(knn_model, knn_model_path)
print(f"KNN model saved successfully.")

# Plot before vs after preprocessing (standardized and normalized)
# Select some random features to plot for comparison

# Here, we select the first 3 features for simplicity (adjust as needed)
features_to_plot = true_data_features.columns[:3]  # Select first 3 columns for plotting

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot before preprocessing (original data)
for i, feature in enumerate(features_to_plot):
    axes[0, i].hist(true_data_features[feature], bins=50, alpha=0.7, label='True', color='blue')
    axes[0, i].hist(false_data_features[feature], bins=50, alpha=0.7, label='False', color='red')
    axes[0, i].set_title(f"Before Preprocessing: {feature}")
    axes[0, i].legend()

# Plot after preprocessing (standardized and normalized)
for i, feature in enumerate(features_to_plot):
    axes[1, i].hist(true_data_standardized_df[feature], bins=50, alpha=0.7, label='True (Standardized)', color='blue')
    axes[1, i].hist(false_data_standardized_df[feature], bins=50, alpha=0.7, label='False (Standardized)', color='red')
    axes[1, i].set_title(f"After Preprocessing (Standardized): {feature}")
    axes[1, i].legend()

# Adjust layout
plt.tight_layout()

# Save the comparison plot
comparison_plot_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/plots/before_vs_after_preprocessing.png'
plt.savefig(comparison_plot_path)
print(f"\nComparison plot saved as {comparison_plot_path}")

plt.show()
