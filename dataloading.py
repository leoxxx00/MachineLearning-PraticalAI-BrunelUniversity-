# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script to load, analyze, and compare datasets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
directory_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/'  # Directory containing datasets
plots_dir = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/plots'       # Directory to save plots

# Validate the directory path
if not os.path.isdir(directory_path):
    raise NotADirectoryError(f"The path {directory_path} is not a valid directory. Please specify a valid directory.")

# Create plot directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# List all files in the directory
files = [f for f in os.listdir(directory_path) if f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.txt')]

if not files:
    print("No supported data files found in the directory.")
else:
    dataframes = []
    file_details = []

    for file in files:
        file_path = os.path.join(directory_path, file)

        # Check the file extension and read accordingly
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file.endswith('.txt'):
                df = pd.read_csv(file_path, delimiter='\t')  # Adjust delimiter if needed
            else:
                print(f"Unsupported file format for {file}")
                continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        # Print file name
        print(f"\nDetails for file: {file}")

        # Print the first few rows
        print("\nFirst 5 rows:")
        print(df.head())

        # Print the data summary
        print("\nData Information:")
        df.info()

        # Print feature summary
        feature_summary = df.describe()
        print("\nFeature Summary:")
        print(feature_summary)

        # Save feature summary to CSV
        summary_filename = f"{os.path.splitext(file)[0]}_feature_summary.csv"
        summary_path = os.path.join(plots_dir, summary_filename)
        feature_summary.to_csv(summary_path)
        print(f"Feature summary saved as {summary_path}")

        # Plot and save the correlation matrix
        if not df.empty:
            # Select numeric columns only
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty:
                try:
                    correlation = numeric_df.corr()  # Calculate correlation matrix
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
                    plt.title(f"Correlation Matrix for {file}")
                    
                    # Save the plot
                    correlation_plot_filename = f"{os.path.splitext(file)[0]}_correlation_matrix.png"
                    correlation_plot_path = os.path.join(plots_dir, correlation_plot_filename)
                    plt.savefig(correlation_plot_path)
                    print(f"Correlation matrix plot saved as {correlation_plot_path}")
                    plt.close()
                except Exception as e:
                    print(f"Error plotting correlation matrix for {file}: {e}")
            else:
                print(f"No numeric columns available for correlation matrix in {file}")

        # Print column names
        print("\nColumn Names:")
        print(df.columns.tolist())

        # Print entire dataset (optional, comment out for large datasets)
        print("\nEntire Dataset:")
        print(df)

        # Store dataframe and details for comparison
        dataframes.append(df)
        file_details.append({
            'file_name': file,
            'num_rows': df.shape[0],
            'num_columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict()
        })

    # Compare details between datasets
    print("\nComparison of Loaded Datasets:")
    for i in range(len(file_details) - 1):
        for j in range(i + 1, len(file_details)):
            file1 = file_details[i]
            file2 = file_details[j]

            print(f"\nComparing {file1['file_name']} and {file2['file_name']}:")
            # Compare number of columns
            if file1['num_columns'] == file2['num_columns']:
                print("- Both have the same number of columns.")
            else:
                print(f"- Different number of columns: {file1['num_columns']} vs {file2['num_columns']}")

            # Compare column names
            common_columns = set(file1['column_names']).intersection(set(file2['column_names']))
            if common_columns:
                print(f"- Common columns: {common_columns}")
            else:
                print("- No common columns.")

            # Compare data types
            same_dtypes = all(file1['dtypes'].get(col) == file2['dtypes'].get(col) for col in common_columns)
            if same_dtypes:
                print("- Common columns have the same data types.")
            else:
                print("- Some common columns have different data types.")
