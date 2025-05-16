import pandas as pd
import numpy as np

# Load the CSV file into a pandas DataFrame
try:
    df = pd.read_csv("Ressources/results-survey539458_16052025.csv")

    # Drop rows where 'Response ID' is NaN, as these are likely incomplete/test entries
    if 'Response ID' in df.columns:
        df.dropna(subset=['Response ID'], inplace=True)
        try:
            df['Response ID'] = pd.to_numeric(df['Response ID'], errors='coerce')
            df.dropna(subset=['Response ID'], inplace=True)
            df['Response ID'] = df['Response ID'].astype(int)
        except Exception:
            pass  # If it fails, proceed without this specific conversion

    # --- Data Cleaning and Conversion ---
    df_original_for_mapping = pd.read_csv("Ressources/results-survey539458_16052025.csv")
    if 'Response ID' in df_original_for_mapping.columns:  # Apply the same initial cleaning
        df_original_for_mapping.dropna(subset=['Response ID'], inplace=True)
        try:
            df_original_for_mapping['Response ID'] = pd.to_numeric(df_original_for_mapping['Response ID'],
                                                                   errors='coerce')
            df_original_for_mapping.dropna(subset=['Response ID'], inplace=True)
            df_original_for_mapping['Response ID'] = df_original_for_mapping['Response ID'].astype(int)
        except Exception:
            pass

    # Align indices if necessary, especially after dropping NaNs
    df = df.set_index('Response ID', drop=False)  # Assuming Response ID is unique and suitable as index
    df_original_for_mapping = df_original_for_mapping.set_index('Response ID', drop=False)
    df_original_for_mapping = df_original_for_mapping.reindex(df.index)

    likert_mapping_detailed = {
        "1 - Not at all": 1, "Not at all": 1,
        "1 - Strongly disagree": 1, "Strongly disagree": 1,
        "1 - Very infrequently": 1, "Very infrequently": 1,
        "1 - Never": 1, "Never": 1,
        "2": 2,
        "2 - Disagree": 2, "Disagree": 2,
        "2 - Infrequently": 2, "Infrequently": 2,
        "3": 3,
        "3 - Neutral": 3, "Neutral": 3,
        "3 - Sometimes": 3, "Sometimes": 3,
        "3 - Neither agree nor disagree": 3, "Neither agree nor disagree": 3,
        "4": 4,
        "4 - Agree": 4, "Agree": 4,
        "4 - Frequently": 4, "Frequently": 4,
        "5": 5,
        "5 - A great extent": 5, "A great extent": 5,
        "5 - Strongly agree": 5, "Strongly agree": 5,
        "5 - Very frequently": 5, "Very frequently": 5,
        "5 - Always": 5, "Always": 5
    }

    binary_mapping = {
        "Yes": 1, "No": 0,
        "Full-time": 1, "Part-time": 0,
        "Male": 0, "Female": 1,  # Example: ensure this matches your data's representation
        # Add other binary mappings as needed
    }

    converted_columns_log = []

    for col in df.columns:
        original_col_data = df_original_for_mapping[col]  # Use data from the freshly loaded mapping DF

        if original_col_data.dtype == 'object':
            # Attempt Likert mapping
            mapped_values = original_col_data.map(likert_mapping_detailed)

            # Attempt binary mapping if not covered by Likert
            # Only apply if mapped_values doesn't have many hits, or do it selectively
            # For simplicity here, we assume Likert is primary for object cols that look like it.
            # A more robust way is to check unique values of original_col_data.

            # If any values were mapped by Likert scale:
            if not mapped_values.isnull().all():
                df[col] = pd.to_numeric(mapped_values, errors='coerce')
                if not df[col].isnull().all():
                    converted_columns_log.append(f"{col} (mapped as Likert)")
            else:  # If likert mapping did nothing, try binary or just plain numeric conversion
                mapped_binary = original_col_data.map(binary_mapping)
                if not mapped_binary.isnull().all():
                    df[col] = pd.to_numeric(mapped_binary, errors='coerce')
                    if not df[col].isnull().all():
                        converted_columns_log.append(f"{col} (mapped as binary)")
                else:
                    # Final attempt: direct to_numeric for object columns not mapped
                    df[col] = pd.to_numeric(original_col_data, errors='coerce')

        else:  # If already numeric, ensure it is so in the final df
            df[col] = pd.to_numeric(original_col_data, errors='coerce')

    # --- Correlation Analysis ---
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.dropna(axis=1, how='all')

    cols_to_drop_from_corr = ['Response ID', 'Seed', 'Last page']
    # Ensure columns to drop actually exist before trying to drop them
    cols_exist_in_numeric_df = [col for col in cols_to_drop_from_corr if col in numeric_df.columns]
    if cols_exist_in_numeric_df:
        numeric_df = numeric_df.drop(columns=cols_exist_in_numeric_df)

    # Reset index if 'Response ID' was used as index and is now a column to be dropped (or was already dropped)
    if 'Response ID' not in numeric_df.columns and 'Response ID' in df.columns:
        # This case implies 'Response ID' might have been index and then numeric_df was created.
        # If 'Response ID' was the index of df, select_dtypes might not pick it up if it's an Int64Index etc.
        # This part might need adjustment based on whether Response ID is kept as index or column.
        # For simplicity, if Response ID was the index of df, and we are creating numeric_df,
        # it won't be part of numeric_df.columns unless explicitly added.
        pass

    if numeric_df.empty or numeric_df.shape[1] < 2:
        print("Not enough numeric data to perform correlation analysis after cleaning.")
        print(f"Number of numeric columns found: {numeric_df.shape[1]}")
        if not numeric_df.empty:
            print(f"Numeric columns: {numeric_df.columns.tolist()}")
    else:
        correlation_matrix = numeric_df.corr()
        print("\nCorrelation Matrix (after cleaning and conversion):")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        print(correlation_matrix)

        if converted_columns_log:
            print("\n--- Columns with attempted mapping to numeric: ---")
            for entry in converted_columns_log:
                print(entry)

        print(f"\nShape of the data used for correlation: {numeric_df.shape}")
        print(f"Columns included in correlation: {numeric_df.columns.tolist()}")

        if correlation_matrix.shape[0] > 10:
            print("\n---")
            print("The correlation matrix is quite large.")
            print("For better visualization, consider using a heatmap, for example with seaborn:")
            print("import seaborn as sns")
            print("import matplotlib.pyplot as plt")
            print("plt.figure(figsize=(20, 18)) # Adjust size as needed")
            print(
                "sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f') # annot=True can be slow for large matrices")
            print("plt.title('Correlation Heatmap of Survey Responses')")
            print("# plt.show() # Uncomment to display plot if running locally")
            print("---")

except FileNotFoundError:
    print(f"Error: The file 'results-survey539458_16052025.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    print(traceback.format_exc())