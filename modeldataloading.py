import os
import pandas as pd

# Path to the directory containing the data (not the file itself)
path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/'  # Update this to the directory where your data files are stored

# Check if the given path is a directory
if not os.path.isdir(path):
    print(f"Error: {path} is not a valid directory.")
else:
    # List all files in the directory with supported extensions
    files = [f for f in os.listdir(path) if f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.txt')]

    if not files:
        print("No supported data files found in the directory.")
    else:
        dataframes = []
        file_details = []

        for file in files:
            file_path = os.path.join(path, file)

            # Check the file extension and read accordingly
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file.endswith('.txt'):
                df = pd.read_csv(file_path, delimiter='\t')  # Adjust delimiter if needed
            else:
                print(f"Unsupported file format for {file}")
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
            summary_path = os.path.join(path, f"{file}_feature_summary.csv")
            feature_summary.to_csv(summary_path)
            print(f"Feature summary saved as {summary_path}")

            # Print column names
            print("\nColumn Names:")
            print(df.columns.tolist())

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
