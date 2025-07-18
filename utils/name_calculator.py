# ai-photo-processor/utils/name_calculator.py

import os
import pandas as pd

def _generate_suffixes(group_size: int, mode: str, custom_pattern: list) -> list:
    """Generates a list of suffixes based on the specified mode."""
    suffixes = []
    
    if mode == 'Standard (d, v, d2, v2, ...)':
        pattern = ['d', 'v']
        pattern_len = 2
    elif mode == 'Wing Clips (v1, v2, v3, ...)':
        # Wing clips don't use a pattern, just a counter.
        return [f"v{i+1}" for i in range(group_size)]
    elif mode == 'Custom':
        pattern = custom_pattern
        if not pattern: # Fallback if custom pattern is empty
            return [f"_{i+1}" for i in range(group_size)]
        pattern_len = len(pattern)
    else: # Default/fallback if no mode matches
        return [f"_{i+1}" for i in range(group_size)]

    for i in range(group_size):
        round_num = i // pattern_len
        pattern_idx = i % pattern_len
        suffix = pattern[pattern_idx]
        if round_num > 0:
            suffix += str(round_num + 1)
        suffixes.append(suffix)
        
    return suffixes

def calculate_final_names(df: pd.DataFrame, main_column: str, suffix_mode: str, custom_suffixes_str: str) -> pd.DataFrame:
    """
    Calculates the 'to' and 'suffix' columns based on the main identifier column
    and the selected suffixing rule.
    """
    if df.empty or main_column not in df.columns:
        return df

    df_copy = df.copy()
    df_copy['to'] = ''
    df_copy['suffix'] = ''

    # Parse the custom suffix string into a list of non-empty strings
    custom_pattern = [s.strip() for s in custom_suffixes_str.split(',') if s.strip()]

    # Determine which rows to process
    if 'skip' in df_copy.columns:
        process_mask = df_copy['skip'] != 'x'
    else:
        process_mask = df_copy[main_column].notna() & (df_copy[main_column] != '')
    
    # Group by the main identifier for the rows that need naming
    for identifier, group in df_copy[process_mask].groupby(main_column, sort=False):
        if not identifier or pd.isna(identifier):
            continue
           
        # Generate suffixes for the group based on the selected mode
        suffixes = _generate_suffixes(len(group), suffix_mode, custom_pattern)

        # Apply the new names and suffixes to the relevant indices
        for i, suffix in enumerate(suffixes):
            df_index = group.index[i]
            df_copy.at[df_index, 'suffix'] = suffix
            
            base_name = f"{identifier}{suffix}"
            original_path = df_copy.at[df_index, 'from']
            _, original_extension = os.path.splitext(original_path)
            
            df_copy.at[df_index, 'to'] = f"{base_name}{original_extension}"
           
    return df_copy