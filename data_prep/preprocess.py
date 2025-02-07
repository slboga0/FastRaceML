import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def prepare_data_for_modeling(df, field_map, target_col="res_finish_position"):
    """
    Prepare data for modeling by selecting predictive columns,
    one-hot encoding categorical features, and splitting the data.
    """
    predictive_cols = [f["field_name"] for f in field_map if f["predictive_feature"] == 1]
    if not predictive_cols:
        raise ValueError("No predictive columns found in field_map.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df[predictive_cols].copy()
    y = df[target_col].copy()

    # One-hot encode categorical features.
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split data into train (60%), validation (20%), and test (20%).
    X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
