import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

try:
    df_all = pd.read_csv("Ressources/results-survey539458_qCode.csv")

    df_completed = df_all[df_all["lastpage"] == 10]
    df_completed = df_completed.dropna(axis=1, how = 'all')

    df_completed.replace(r'^\s*(\d+)\s*-.*$', r'\1', regex=True, inplace=True)
    df = df_completed.apply(pd.to_numeric, errors='ignore')

    df.columns = [f"{i+1}: {col}" for i, col in enumerate(df.columns)]

    output_path = "Ressources/processed_survey_qCode.csv"
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")

    frequency_map = {
        "never": 1,
        "rarely": 2,
        "sometimes": 3,
        "often": 4,
        "always": 5
    }

    # Apply frequency mapping to selected columns (columns 68 to 84, inclusive)
    target_columns = df.columns[67:84]  # Python uses 0-based indexing
    df[target_columns] = df[target_columns].map(frequency_map.get)

    #41: UseQ01[SQ001]

    frequency_map = {
        "Never": 1,
        "Once a month": 2,
        "Multiple times a month": 3,
        "Once a week": 4,
        "Multiple times a week": 5,
        "Every day": 6,
        "Multiple times a day": 7,
        "Every time I need help": 8
    }

    df["41: UseQ01[SQ001]"] = df["41: UseQ01[SQ001]"].map(frequency_map.get)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Compute correlation matrix
    correlation_matrix = numeric_df.corr(method='pearson')

    # Optionally display it
    print(correlation_matrix)

    # (Optional) Save to CSV
    correlation_matrix.to_csv("Ressources/correlation_matrix.csv")

    # Identify strongly correlated pairs (absolute correlation > 0.6 and not 1.0)
    threshold = 0.5
    corr_pairs = correlation_matrix.unstack()
    strong_corr = corr_pairs[(abs(corr_pairs) > threshold) & (abs(corr_pairs) < 1.0)]
    strong_corr = strong_corr.drop_duplicates().sort_values(ascending=False)

    # Convert to DataFrame and save
    strong_corr_df = strong_corr.reset_index()
    strong_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
    strong_corr_df.to_csv("Ressources/strong_correlations.csv", index=False)

    print("Strongly correlated variable pairs saved to 'Ressources/strong_correlations.csv'")
except FileNotFoundError:
    print(f"Error: The file 'Ressources/results-survey539458_qCode.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    print(traceback.format_exc())