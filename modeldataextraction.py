import pandas as pd

# Define the file path
file_path =  '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/data.csv'

# Define the output file paths
filtered_file_path =  '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/data.csv'
true_file_path =  '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/true_data.csv'
false_file_path =  '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/false_data.csv'

# Load the data and handle potential errors
try:
    # Load as CSV
    print("Attempting to load the file as a CSV...")
    data = pd.read_csv(file_path)
    print("Data Loaded Successfully (CSV):")
    print(data.head())
except pd.errors.ParserError:
    # If not a CSV, try loading as Excel
    try:
        print("Failed to load as CSV. Attempting to load the file as an Excel spreadsheet...")
        data = pd.read_excel(file_path)
        print("Data Loaded Successfully (Excel):")
        print(data.head())
    except Exception as excel_error:
        print("Failed to load Excel file:", excel_error)
        data = None
except FileNotFoundError:
    print(f"File not found at: {file_path}")
    data = None
except Exception as e:
    print("An error occurred while loading the file:", e)
    data = None

# If data is successfully loaded, process it
if data is not None:
    try:
        print("\nFiltering the dataset to include only 'co', 'smoke', 'gas', and 'motion' columns...")

        # Select the specified columns
        filtered_data = data[['co', 'smoke', 'gas', 'motion']]
        print("Filtered Data:")
        print(filtered_data.head())

        # Save the filtered data to a new CSV file
        print(f"\nSaving the filtered data to {filtered_file_path}...")
        filtered_data.to_csv(filtered_file_path, index=False)
        print(f"Filtered data saved successfully to {filtered_file_path}")

        # Split the data based on 'motion' column
        print("\nSplitting the filtered data into 'motion = True' and 'motion = False' subsets...")
        true_data = filtered_data[filtered_data['motion'] == True]
        false_data = filtered_data[filtered_data['motion'] == False]

        # Save each subset to separate files
        print(f"Saving 'motion = True' data to {true_file_path}...")
        true_data.to_csv(true_file_path, index=False)
        print(f"'motion = True' data saved successfully to {true_file_path}")

        print(f"Saving 'motion = False' data to {false_file_path}...")
        false_data.to_csv(false_file_path, index=False)
        print(f"'motion = False' data saved successfully to {false_file_path}")

    except KeyError as e:
        print(f"One or more columns are missing in the dataset: {e}")
    except Exception as e:
        print(f"An error occurred while filtering, splitting, or saving the data: {e}")
