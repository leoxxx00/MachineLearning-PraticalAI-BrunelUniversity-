import pandas as pd

true_file_path =  '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/true_data.csv'
false_file_path =  '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/false_data.csv'
balanced_true_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/balanced_true_data.csv'
balanced_false_file_path = '/Users/htet/Desktop/ML/machinelearning/ML/myenv/data/balanced_false_data.csv'

try:
    # Load the 'motion = True' dataset
    print(f"Loading 'motion = True' data from {true_file_path}...")
    true_data = pd.read_csv(true_file_path)
    true_count = len(true_data)
    print(f"'motion = True' data loaded successfully with {true_count} rows.")

    # Load the 'motion = False' dataset
    print(f"Loading 'motion = False' data from {false_file_path}...")
    false_data = pd.read_csv(false_file_path)
    false_count = len(false_data)
    print(f"'motion = False' data loaded successfully with {false_count} rows.")

    # Print counts before balancing
    print(f"\nCounts before balancing:")
    print(f"Rows with 'motion = True': {true_count}")
    print(f"Rows with 'motion = False': {false_count}")

    # Define target count
    target_count = 404702

    # Balance 'motion = True' data by duplicating rows
    if true_count < target_count:
        print(f"\nBalancing 'motion = True' data to match {target_count} rows...")
        repeat_factor = (target_count // true_count) + 1
        balanced_true_data = pd.concat([true_data] * repeat_factor, ignore_index=True).iloc[:target_count]
        print(f"'motion = True' data balanced successfully with {len(balanced_true_data)} rows.")
    else:
        balanced_true_data = true_data

    # Balance 'motion = False' data by trimming rows
    if false_count > target_count:
        print(f"\nTrimming 'motion = False' data to match {target_count} rows...")
        balanced_false_data = false_data.sample(n=target_count, random_state=42)
        print(f"'motion = False' data trimmed successfully with {len(balanced_false_data)} rows.")
    else:
        balanced_false_data = false_data

    # Save the balanced datasets
    print(f"\nSaving balanced 'motion = True' data to {balanced_true_file_path}...")
    balanced_true_data.to_csv(balanced_true_file_path, index=False)
    print(f"Balanced 'motion = True' data saved successfully.")

    print(f"\nSaving balanced 'motion = False' data to {balanced_false_file_path}...")
    balanced_false_data.to_csv(balanced_false_file_path, index=False)
    print(f"Balanced 'motion = False' data saved successfully.")

    # Print counts after balancing
    print(f"\nCounts after balancing:")
    print(f"Rows with 'motion = True': {len(balanced_true_data)}")
    print(f"Rows with 'motion = False': {len(balanced_false_data)}")

except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred while loading, balancing, or saving the data: {e}")