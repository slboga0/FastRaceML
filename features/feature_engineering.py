#!/usr/bin/env python
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def impute_missing_values(df):
    """
    Impute missing values: numeric columns use the median; categorical columns use 'Missing'.
    """
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_imputed.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        median_value = df_imputed[col].median()
        df_imputed[col].fillna(median_value, inplace=True)
    for col in categorical_cols:
        df_imputed[col].fillna("Missing", inplace=True)
    return df_imputed

def engineer_features(merged_df, result_field_map, pp_field_map):
    """
    Build a feature-engineered DataFrame by selecting columns marked as predictive.
    Missing values are imputed, and the result is saved to CSV.
    """
    predictive_fields_result = [
        entry["field_name"] for entry in result_field_map
        if str(entry.get("predictive_feature", "")).strip() == "1"
    ]
    predictive_fields_pp = [
        entry["field_name"] for entry in pp_field_map
        if str(entry.get("predictive_feature", "")).strip() == "1"
    ]
    combined_predictive_fields = list(set(predictive_fields_result + predictive_fields_pp))
    logger.info(f"Combined predictive fields: {combined_predictive_fields}")

    feature_columns = [col for col in merged_df.columns if col in combined_predictive_fields]
    if not feature_columns:
        raise ValueError("No predictive feature columns found in the merged DataFrame.")

    engineered_df = merged_df[feature_columns].copy()
    engineered_df = impute_missing_values(engineered_df)

    output_path = Path("data") / "outputs" / "engineered_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(output_path, index=False)
    logger.info(f"Engineered features saved to '{output_path}'.")
    return engineered_df

if __name__ == "__main__":
    import sys
    import numpy as np
    import pandas as pd
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    merged_data = {
        "res_track_name": ["SA", "SA", "SA", "SA"],
        "res_race_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "res_race_number": [1, 2, 3, 4],
        "res_finish_position": [2, 1, 3, np.nan],
        "track": ["SA", "SA", "SA", "SA"],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "race_no": [1, 2, 3, 4],
        "entry": ["Horse A", "Horse B", "Horse C", "Horse D"]
    }
    merged_df = pd.DataFrame(merged_data)
    
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
