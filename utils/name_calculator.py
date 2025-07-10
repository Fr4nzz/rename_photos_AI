# ai-photo-processor/utils/name_calculator.py

import os
import pandas as pd

def calculate_final_names(df: pd.DataFrame, main_column: str) -> pd.DataFrame:
    """
    Calculates the 'to' column based on the main identifier column and suffixes.
    This implementation assumes a simple pair 'd'/'v' suffixing for each unique
    identifier in the main_column.
    """
    if df.empty or main_column not in df.columns:
        return df

    df_copy = df.copy()
    df_copy['to'] = '' # Ensure 'to' column exists and is cleared

    # Determine which rows to process
    if 'skip' in df_copy.columns:
        process_mask = df_copy['skip'] != 'x'
    else:
        # If no skip column, process all rows that have a main_column value
        process_mask = df_copy[main_column].notna() & (df_copy[main_column] != '')
    
    # Group by the main identifier, but only for the rows we need to process
    for identifier, group in df_copy[process_mask].groupby(main_column, sort=False):
        if not identifier or pd.isna(identifier):
            continue
           
        suffixes = []
        pattern = ['d', 'v']
        for i in range(len(group)):
            round_num = i // 2
            pattern_idx = i % 2
            suffix = pattern[pattern_idx]
            if round_num > 0:
                suffix += str(round_num + 1)
            suffixes.append(suffix)

        # Apply the new names only to the relevant indices
        for i, suffix in enumerate(suffixes):
            df_index = group.index[i]
            base_name = f"{identifier}{suffix}"
           
            original_path = df_copy.at[df_index, 'from']
            # Safely get the original extension
            _, original_extension = os.path.splitext(original_path)
           
            df_copy.at[df_index, 'to'] = f"{base_name}{original_extension}"
           
    return df_copy