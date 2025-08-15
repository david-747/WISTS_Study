import csv
import os


def create_question_map(questions_file_path):
    """
    Reads the questions file and creates a mapping from question code to full question text.
    The question codes and texts are expected to be in the header of the CSV.
    """
    question_map = {}
    try:
        with open(questions_file_path, 'r', encoding='utf-8', newline='') as q_file:
            reader = csv.reader(q_file)
            header = next(reader)  # Read the first line (header)
            for item in header:
                parts = item.split('.', 1)  # Split only on the first period
                if len(parts) == 2:
                    code = parts[0].strip()
                    full_question = parts[1].strip()
                    question_map[code] = full_question
                # else:
                #     print(f"Warning: Could not parse question code/text from header item: {item}")
    except FileNotFoundError:
        print(f"Error: The questions file '{questions_file_path}' was not found.")
        return None
    except StopIteration:
        print(f"Error: The questions file '{questions_file_path}' is empty or has no header.")
        return None
    except Exception as e:
        print(f"Error reading or parsing questions file '{questions_file_path}': {e}")
        return None

    if not question_map:
        print("Warning: No question codes could be mapped from the questions file. Full questions might be missing.")
    return question_map


def modify_correlations_file_in_place(correlations_file_path, question_map_dict):
    """
    Modifies the correlations CSV file in-place.
    It adds two new columns for full question text (at index 3 and 4)
    and shifts subsequent original columns.
    """
    if question_map_dict is None:
        print("Error: Question map is not available. Cannot proceed with file modification.")
        return

    all_original_rows = []
    try:
        with open(correlations_file_path, 'r', encoding='utf-8', newline='') as c_file:
            reader = csv.reader(c_file)
            for row in reader:
                all_original_rows.append(row)

        if not all_original_rows:
            print(f"Error: The correlations file '{correlations_file_path}' is empty.")
            return
    except FileNotFoundError:
        print(f"Error: The correlations file '{correlations_file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading correlations file '{correlations_file_path}': {e}")
        return

    original_header = all_original_rows[0]
    original_data_rows = all_original_rows[1:]

    # Prepare the new header
    if len(original_header) < 3:  # Need at least Var1, Var2, Correlation
        print(
            f"Error: Correlations file '{correlations_file_path}' has an unexpected header format (less than 3 columns). Cannot proceed.")
        return

    # New header: Original_Col0, Original_Col1, Original_Col2, New_FullQ1, New_FullQ2, Original_Col3, Original_Col4, ...
    new_header_list = original_header[:3] + ["Full Question Variable 1", "Full Question Variable 2"]
    if len(original_header) > 3:  # If there were original columns from index 3 onwards
        new_header_list.extend(original_header[3:])

    modified_content_for_writing = [new_header_list]

    for row_idx, row_data in enumerate(original_data_rows, 1):
        # Initialize new_row with empty strings to match the new header's length
        current_new_row = [""] * len(new_header_list)

        # Copy original first 3 columns (Var1, Var2, Correlation)
        for i in range(min(3, len(row_data))):
            current_new_row[i] = row_data[i].strip()

        var1_raw = current_new_row[0]
        var2_raw = current_new_row[1]
        # correlation_val = current_new_row[2] # Already set

        full_q1_text = "N/A - Code processing error"
        full_q2_text = "N/A - Code processing error"

        if var1_raw:  # Attempt to get full question for Var1
            code1 = var1_raw.split(': ')[-1] if ': ' in var1_raw else var1_raw
            full_q1_text = question_map_dict.get(code1, f"N/A - Code '{code1}' not found")
        else:
            full_q1_text = "N/A - Var1 code is empty"

        if var2_raw:  # Attempt to get full question for Var2
            code2 = var2_raw.split(': ')[-1] if ': ' in var2_raw else var2_raw
            full_q2_text = question_map_dict.get(code2, f"N/A - Code '{code2}' not found")
        else:
            full_q2_text = "N/A - Var2 code is empty"

        current_new_row[3] = full_q1_text  # New 4th column
        current_new_row[4] = full_q2_text  # New 5th column

        # Copy original columns that were at index 3 onwards into their new positions (starting at index 5)
        original_col_src_idx = 3  # Start from the original 4th column
        new_col_target_idx = 5  # Start writing into the new 6th column

        while original_col_src_idx < len(row_data) and new_col_target_idx < len(current_new_row):
            current_new_row[new_col_target_idx] = row_data[original_col_src_idx].strip()
            original_col_src_idx += 1
            new_col_target_idx += 1

        modified_content_for_writing.append(current_new_row)

    # Now write the modified content back to the original file
    try:
        with open(correlations_file_path, 'w', newline='', encoding='utf-8') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(modified_content_for_writing)
        print(f"\nSuccessfully modified file: '{correlations_file_path}'")
        print("The 4th column is now 'Full Question Variable 1'.")
        print("The 5th column is now 'Full Question Variable 2'.")
        if len(original_header) > 3 and len(new_header_list) > 5:
            print(
                f"The content of the original 4th column (header: '{original_header[3]}') and subsequent columns have been shifted to start from the 6th column (header: '{new_header_list[5]}').")
        elif len(original_header) == 3:
            print("The file originally had 3 columns. Two new columns for full questions have been added.")

    except Exception as e:
        print(f"\nError writing modified content to '{correlations_file_path}': {e}")
        print("Your original file might be unchanged or partially modified if an error occurred during writing.")


# --- Main execution block ---
if __name__ == "__main__":
    correlations_input_file = "strong_correlations_edited.csv"
    questions_input_file = "results-survey539458_qCode_qText.csv"

    print("----------------------------------------------------------------------")
    print("IMPORTANT: This script will modify the file")
    print(f"'{correlations_input_file}' IN-PLACE.")
    print("It is STRONGLY recommended to MAKE A BACKUP of this file before")
    print("proceeding to avoid accidental data loss.")
    print("----------------------------------------------------------------------")

    # In a real interactive script, you might uncomment the next line:
    # input("Press Enter to continue or Ctrl+C to abort...")

    print(f"\nAttempting to read question map from: '{questions_input_file}'...")
    q_map = create_question_map(questions_input_file)

    if q_map is not None:  # Proceed only if question map was created (even if empty)
        print(f"Question map created. Number of entries: {len(q_map)}")
        print(f"\nAttempting to modify: '{correlations_input_file}'...")
        modify_correlations_file_in_place(correlations_input_file, q_map)
    else:
        print("\nAborting modification due to critical issues with creating the question map.")
