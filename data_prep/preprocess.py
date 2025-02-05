# FastRaceML/data_prep/preprocess.py

"""
Preprocessing utilities for splitting data, handling missing values, 
and preparing for model training.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def prepare_data_for_modeling(df, field_map, target_col="res_finish_position"):
    """
    - Select columns based on field_map['predictive_feature'] == 1
    - Remove rows with NaN in those columns or the target
    - One-hot encode
    - Split into train, validation, and test sets
    """
    predictive_cols = [f["field_name"] for f in field_map if f["predictive_feature"] == 1]
    
    # Basic checks
    if not predictive_cols:
        raise ValueError("No predictive columns found in field_map.")

    print (df.columns)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")

    X = df[predictive_cols].copy()
    y = df[target_col].copy()

    # Drop rows with NA
    # valid_rows = X.notna().all(axis=1) & y.notna()
    #X = X[valid_rows]
    #y = y[valid_rows]

    #if X.empty:
    #    raise ValueError("No valid data after dropping missing values in features/target.")

    # One-hot encode
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train/Val/Test split: 60%/20%/20% (or 70%/15%/15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test
