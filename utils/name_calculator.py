# ai-photo-processor/utils/name_calculator.py

import os
import pandas as pd
from typing import Dict, Any

def _get_suffixes(num_items: int, mode: str, custom_pattern: str) -> list[str]:
    """Generates a list of suffixes based on the selected mode."""
    suffixes = []
    
    if mode == 'Wing Clips':
        return [f"v{i+1}" for i in range(num_items)]

    if mode == 'Custom':
        pattern = [s.strip() for s in custom_pattern.split(',') if s.strip()]
        if not pattern: # Fallback to standard if custom is empty
            mode = 'Standard'
    
    if mode == 'Standard':
        pattern = ['d', 'v']

    # Logic for Standard and Custom modes (cycling with numbers)
    for i in range(num_items):
        cycle = i // len(pattern)
        base_suffix = pattern[i % len(pattern)]
        suffixes.append(f"{base_suffix}{cycle + 1}" if cycle > 0 else base_suffix)
        
    return suffixes

def calculate_final_names(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculates the 'to' and 'suffix' columns based on the main identifier column
    and the suffixing rules defined in the settings.
    """
    main_column = settings.get('main_column', 'CAM')
    if df.empty or main_column not in df.columns:
        return df

    df_copy = df.copy()
    df_copy['to'] = ''
    df_copy['suffix'] = ''

    process_mask = (df_copy['skip'] != 'x') if 'skip' in df_copy.columns else (df_copy[main_column].notna() & (df_copy[main_column] != ''))
    
    # Group by the main identifier for the rows that need processing
    for identifier, group in df_copy[process_mask].groupby(main_column, sort=False):
        if not identifier or pd.isna(identifier):
            continue
        
        # Generate suffixes for the number of items in this group
        group_suffixes = _get_suffixes(
            num_items=len(group),
            mode=settings.get('suffix_mode', 'Standard'),
            custom_pattern=settings.get('custom_suffixes', 'd,v')
        )

        # Apply the new names and suffixes to the original dataframe indices
        for i, (df_index, _) in enumerate(group.iterrows()):
            suffix = group_suffixes[i]
            df_copy.at[df_index, 'suffix'] = suffix
            
            base_name = f"{identifier}{suffix}"
            original_path = df_copy.at[df_index, 'from']
            _, original_extension = os.path.splitext(original_path)
            
            df_copy.at[df_index, 'to'] = f"{base_name}{original_extension}"
            
    return df_copy