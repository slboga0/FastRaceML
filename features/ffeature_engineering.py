#!/usr/bin/env python
"""
feature_engineering.py

This module defines functions to perform feature engineering on a merged DataFrame.
It selects only those columns whose corresponding entry in either the result field mapping 
or the past performance (PP) field mapping has predictive_feature equal to "1".
It also provides a simple imputation step to handle missing values.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def impute_missing_values(df):
    """
    Imputes missing values in the DataFrame.
    
    - For numeric columns, fills missing values with the median.
    - For non-numeric (categorical) columns, fills missing values with the string "Missing".
    
    Args:
        df (pd.DataFrame): DataFrame with missing values.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df_imputed = df.copy()
    
    # Identify numeric and non-numeric columns.
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_imputed.select_dtypes(exclude=[np.number]).columns
    
    # Impute numeric columns with median.
    for col in numeric_cols:
        median_value = df_imputed[col].median()
        df_imputed[col].fillna(median_value, inplace=True)
        logger.info(f"Imputed missing values in numeric column '{col}' with median: {median_value}")
    
    # Impute non-numeric columns with a placeholder string.
    for col in categorical_cols:
        df_imputed[col].fillna("Missing", inplace=True)
        logger.info(f"Imputed missing values in categorical column '{col}' with 'Missing'")
    
    return df_imputed

def engineer_features(merged_df, result_field_map, pp_field_map):
    """
    Returns a feature-engineered DataFrame that includes only those columns from the merged DataFrame
    for which the corresponding entry in either the result field mapping or the PP field mapping 
    has predictive_feature equal to "1". Missing values are imputed instead of dropped.
    
    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing PP and result fields.
        result_field_map (list of dict): Mapping for result fields.
        pp_field_map (list of dict): Mapping for past performance fields.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only the combined predictive features with missing values imputed.
    """
    # Extract predictive fields from the result mapping.
    predictive_fields_result = [
        entry["field_name"] for entry in result_field_map
        if str(entry.get("predictive_feature", "")).strip() == "1"
    ]
    logger.info(f"Predictive fields from result mapping: {predictive_fields_result}")
    
    # Extract predictive fields from the PP mapping.
    predictive_fields_pp = [
        entry["field_name"] for entry in pp_field_map
        if str(entry.get("predictive_feature", "")).strip() == "1"
    ]
    logger.info(f"Predictive fields from PP mapping: {predictive_fields_pp}")
    
    # Combine predictive fields from both mappings (remove duplicates).
    combined_predictive_fields = list(set(predictive_fields_result + predictive_fields_pp))
    logger.info(f"Combined predictive fields: {combined_predictive_fields}")
    
    # Select columns from merged_df that match the combined predictive fields.
    feature_columns = [col for col in merged_df.columns if col in combined_predictive_fields]
    logger.info(f"Feature columns found in merged DataFrame: {feature_columns}")
    
    if not feature_columns:
        raise ValueError("No predictive feature columns found in the merged DataFrame.")
    
    # Create the engineered DataFrame from the selected columns.
    engineered_df = merged_df[feature_columns].copy()
    
    # Impute missing values instead of dropping them.
    engineered_df = impute_missing_values(engineered_df)
    logger.info(f"Engineered DataFrame shape after imputation: {engineered_df.shape}")
    
    # Save the engineered DataFrame to CSV.
    engineered_df.to_csv("data\\outputs\\engineered_features.csv", index=False)
    logger.info("Engineered features saved to 'engineered_features.csv'.")

    return engineered_df

# Example usage for testing:
if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Create a dummy merged DataFrame.
    merged_data = {
        "res_track_name": ["SA", "SA", "SA", "SA"],
        "res_race_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "res_race_number": [1, 2, 3, 4],
        "res_finish_position": [2, 1, 3, np.nan],  # One missing value
        "track": ["SA", "SA", "SA", "SA"],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "race_no": [1, 2, 3, 4],
        "entry": ["Horse A", "Horse B", "Horse C", "Horse D"]
    }
    merged_df = pd.DataFrame(merged_data)
    
    # Create dummy field mappings.
    result_field_map = [
        {"field_name": "res_track_name", "predictive_feature": "1"},
        {"field_name": "res_race_date", "predictive_feature": "1"},
        {"field_name": "res_race_number", "predictive_feature": "1"},
        {"field_name": "res_finish_position", "predictive_feature": "1"},
        {"field_name": "res_horse_name", "predictive_feature": "0"}
    ]
    pp_field_map = [
        {"field_name": "track", "predictive_feature": "1"},
        {"field_name": "date", "predictive_feature": "1"},
        {"field_name": "race_no", "predictive_feature": "1"},
        {"field_name": "entry", "predictive_feature": "0"}
    ]
    
    engineered_df = engineer_features(merged_df, result_field_map, pp_field_map)
    print("Engineered features DataFrame:")
    print(engineered_df)
