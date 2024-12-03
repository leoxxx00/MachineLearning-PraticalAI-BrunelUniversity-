import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the balanced datasets
balanced_true_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/preprocessed_true_data.csv'
balanced_false_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/preprocessed_false_data.csv'

try:
    # Load balanced datasets
    print("Loading balanced 'motion = True' data...")
    balanced_true_data = pd.read_csv(balanced_true_file_path)
    print(f"Balanced 'motion = True' data loaded successfully with {len(balanced_true_data)} rows.")

    print("Loading balanced 'motion = False' data...")
    balanced_false_data = pd.read_csv(balanced_false_file_path)
    print(f"Balanced 'motion = False' data loaded successfully with {len(balanced_false_data)} rows.")

    # Combine the datasets
    balanced_true_data['motion'] = 1  # Label 'motion = True' as 1
    balanced_false_data['motion'] = 0  # Label 'motion = False' as 0
    combined_data = pd.concat([balanced_true_data, balanced_false_data], ignore_index=True)

    # Data exploration
    print("\nCombined Dataset Info:")
    print(combined_data.info())
    print("\nCombined Dataset Description:")
    print(combined_data.describe())

    print("\nClass distribution:")
    print(combined_data['motion'].value_counts())

    # Handle missing values
    combined_data = combined_data.dropna()

    # Separate features and target
    X = combined_data.drop(columns=['motion'])  # Features
    y = combined_data['motion']  # Target

    # Print input/output shapes
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Data preprocessing completed.")

    # Define the KNN model
    knn = KNeighborsClassifier()

    # Define hyperparameter grid
    param_grid = {'n_neighbors': range(1, 21)}

    # Perform grid search with cross-validation
    print("Performing grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best model
    best_knn = grid_search.best_estimator_
    print(f"Best KNN parameters: {grid_search.best_params_}")

    # Evaluate the model
    print("\nEvaluating the model...")
    y_pred = best_knn.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate and print weighted F1 score
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1 Score: {weighted_f1}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print final model input/output shape sizes
    print(f"Final Model Input Shape: {X_train.shape}")
    print(f"Final Model Output Shape: {y_train.shape}")

except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
