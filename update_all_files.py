#!/usr/bin/env python
"""
update_all_files.py

This script updates (or creates) all the FastRaceML Python files with the refactored content.
Run it from the project root directory.
"""

import os
from pathlib import Path

# Mapping of relative file paths to their new contents.
files_to_update = {
    "config/HorseRaceParams.py": r'''import os
from data_prep.utilities import setup_logging

# Define directories and constants.
CURRENT_DIR = os.getcwd()
DATA_DIR = os.path.join(CURRENT_DIR, "data")
LOG_DIR = os.path.join(CURRENT_DIR, "logs")
OUTPUT_DIR = os.path.join(DATA_DIR, "predictions")
TEMP_DIR = os.path.join(CURRENT_DIR, "temp")

# Optionally, initialize logging here.
# setup_logging(os.path.join(LOG_DIR, "FastRaceML.log"))

# Additional constants.
MAX_NPP = 10
MAX_NWRK = 12
''',

    "config/__init__.py": r'''# Empty __init__.py for config package
''',

    "data_prep/ingestion.py": r'''import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def scan_cards(root_dir, track_pattern="*.DRF"):
    """
    Scan the specified directory (non-recursively) for files matching the track_pattern
    and return a list of dictionaries for each found file.
    """
    root_path = Path(str(root_dir).strip())
    if not root_path.exists():
        raise FileNotFoundError(f"Directory '{root_path}' does not exist.")

    files = list(root_path.glob(track_pattern))
    logger.info(f"Found {len(files)} file(s) matching '{track_pattern}' in '{root_path}'.")

    card_list = [{"card": file.stem, "filename": file} for file in files]
    return card_list

def load_results_and_merge(pp_all_df, results_dir, result_field_mapping_csv, track_pattern="",
                           result_merge_cols=None):
    """
    Load result CSV files using a field mapping CSV, merge with the past performance DataFrame,
    and save the engineered and merged data.
    """
    if result_merge_cols is None:
        result_merge_cols = {
            "track": "res_track_name",
            "date": "res_race_date",
            "race_no": "res_race_number",
            "horse_name": "res_horse_name"
        }

    mapping_df = pd.read_csv(result_field_mapping_csv, skiprows=1, header=None,
                               names=["field_position", "field_name", "predictive_feature"])
    mapping_df["field_position"] = mapping_df["field_position"].astype(int)
    field_map = mapping_df.sort_values(by="field_position").to_dict(orient="records")

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory '{results_path}' does not exist.")

    all_results = []
    logger.info(f"Loading result files from '{results_path}' using pattern '{track_pattern}'")

    for csv_file in results_path.glob(track_pattern):
        if csv_file.is_file():
            logger.debug(f"Parsing result file: {csv_file}")
            with csv_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items = line.split(",")
                    record = {}
                    for fm in field_map:
                        pos = int(fm["field_position"])
                        col_name = fm["field_name"]
                        record[col_name] = items[pos].strip('"') if pos < len(items) else ""
                    all_results.append(record)
        else:
            logger.debug(f"Skipping non-file: {csv_file}")

    res_columns = [fm["field_name"] for fm in field_map]
    result_df = pd.DataFrame(all_results, columns=res_columns)

    # Save result CSV.
    result_output_path = Path("data") / "outputs" / "results.csv"
    result_output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_output_path, index=False)
    logger.info(f"Engineered features saved to '{result_output_path}'.")

    # Construct merge keys for PP data.
    required_cols = {"track", "date", "race_no", "horse_name"}
    if required_cols.issubset(pp_all_df.columns):
        pp_all_df["race_key"] = (
            pp_all_df["track"].astype(str).str.upper().str.strip() + "_" +
            pp_all_df["date"].astype(str).str.strip() + "_" +
            pp_all_df["race_no"].astype(str).str.strip() + "_" +
            pp_all_df["horse_name"].astype(str).str.upper().str.strip()
        )
    else:
        logger.warning("PP DataFrame missing one or more required columns: 'track', 'date', 'race_no', 'horse_name'")

    missing_merge_cols = [col for col in result_merge_cols.values() if col not in result_df.columns]
    if missing_merge_cols:
        logger.error(f"Result DataFrame is missing merge columns: {missing_merge_cols}")
    else:
        result_df["race_key"] = (
            result_df[result_merge_cols["track"]].astype(str).str.upper().str.strip() + "_" +
            result_df[result_merge_cols["date"]].astype(str).str.strip() + "_" +
            result_df[result_merge_cols["race_no"]].astype(str).str.strip() + "_" +
            result_df[result_merge_cols["horse_name"]].astype(str).str.upper().str.strip()
        )

    logger.info("Merging PP data with result data...")
    if "race_key" in pp_all_df.columns and "race_key" in result_df.columns:
        merged_df = pd.merge(pp_all_df, result_df, how='left', on='race_key')
        merged_df.drop_duplicates(subset=["race_key"], keep="first", inplace=True)
    else:
        logger.warning("Missing 'race_key' in one or both DataFrames; merge cannot be performed properly.")
        merged_df = pp_all_df.copy()

    all_cols = list(merged_df.columns)
    pp_cols_order = [col for col in all_cols if not col.startswith("res_") and col != "race_key"]
    res_cols_order = [col for col in all_cols if col.startswith("res_") and col != "race_key"]
    new_column_order = ["race_key"] + pp_cols_order + res_cols_order
    merged_df = merged_df[new_column_order]

    merged_output_path = Path("data") / "outputs" / "merged_data.csv"
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_output_path, index=False)
    logger.info(f"Merged data saved to '{merged_output_path}'.")

    return merged_df, field_map
''',

    "data_prep/preprocess.py": r'''import logging
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
''',

    "data_prep/readers.py": r'''import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def load_single_card(pp_path, field_map):
    """
    Read a single past performance file and return a list of dictionaries.
    Each dictionary maps a field name (from field_map) to its corresponding value.
    """
    all_pp_list = []
    try:
        with open(pp_path, "r") as f:
            for line in f:
                values = line.strip().split(',')
                record = {
                    fm["field_name"]: values[idx].strip('"') if idx < len(values) else ""
                    for idx, fm in enumerate(field_map)
                }
                all_pp_list.append(record)
    except FileNotFoundError:
        logger.error(f"File not found: {pp_path}")
    except Exception as e:
        logger.error(f"Error reading {pp_path}: {e}")
    return all_pp_list

def load_all_pp_cards(pp_cards, field_mapping_csv):
    """
    Load all past performance cards and concatenate them into a DataFrame.
    """
    mapping_df = pd.read_csv(field_mapping_csv)
    field_map = mapping_df[['field_name', 'predictive_feature']].to_dict(orient='records')

    all_pps = []
    for card in pp_cards:
        filename = card.get("filename")
        if not filename:
            logger.warning(f"Skipping card with missing filename: {card}")
            continue
        single_card_list = load_single_card(filename, field_map)
        all_pps.extend(single_card_list)

    pp_all_df = pd.DataFrame(all_pps)
    return pp_all_df, field_map

def load_all_result(results_dir, track_pattern=""):
    """
    Scan a directory for CSV files matching the given pattern and load all records.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory '{results_path}' does not exist.")

    # Placeholder columns (adjust as needed for your CSV structure).
    columns = ["track", "date", "race_no", "entry", "finish", "other_cols"]
    all_results = []

    logger.info(f"Searching for CSV files in '{results_path}' with pattern '{track_pattern}'")
    for csv_file in results_path.rglob(track_pattern):
        if csv_file.is_file():
            logger.debug(f"Reading file: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                all_results.extend(df.to_dict(orient='records'))
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
        else:
            logger.debug(f"Skipping non-file: {csv_file}")

    logger.info(f"Total records loaded: {len(all_results)} from pattern '{track_pattern}'")
    return all_results, columns
''',

    "data_prep/utilities.py": r'''import os
import logging
import configparser
from pathlib import Path

def setup_logging(log_file=None, level=logging.INFO):
    """
    Configure logging for the application. If log_file is specified, logs are written there.
    """
    log_format = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=level, format=log_format)
    else:
        logging.basicConfig(level=level, format=log_format)
    logging.info("Logging configured.")

def load_config():
    """
    Load configuration from 'config/config.ini'.
    """
    config = configparser.ConfigParser()
    config_file_path = Path("config") / "config.ini"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_file_path}")
    config.read(config_file_path)
    return config
''',

    "data_prep/__init__.py": r'''# Empty __init__.py for data_prep package
''',

    "features/feature_calculator.py": r'''import pandas as pd

class FeatureCalculator:
    @staticmethod
    def calculate_best_lifetime_speed(data: pd.Series) -> float:
        speed = data.get('best_bris_speed_life', 0)
        return float(speed) if pd.notna(speed) else 0.0

    @staticmethod
    def calculate_speed_last_race(data: pd.Series) -> float:
        speed = data.get('speed_rating_856', 0)
        return float(speed) if pd.notna(speed) else 0.0
''',

    "features/feature_engineering.py": r'''#!/usr/bin/env python
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
''',

    "features/feature_set.py": r'''class HorseRaceFeature:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.value = None

    def set_value(self, val):
        self.value = val

    def get_value(self):
        return self.value

class HorseRaceFeatureSet:
    def __init__(self):
        self.features = {}

    def add_feature(self, feature_name):
        if feature_name in self.features:
            raise ValueError(f"Feature '{feature_name}' already exists.")
        self.features[feature_name] = HorseRaceFeature(feature_name)

    def set_feature_value(self, feature_name, val):
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found.")
        self.features[feature_name].set_value(val)

    def get_feature_value(self, feature_name):
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found.")
        return self.features[feature_name].get_value()

    def get_all_features(self):
        return {name: feature.get_value() for name, feature in self.features.items()}
''',

    "features/__init__.py": r'''# Empty __init__.py for features package
''',

    "main/pipeline.py": r'''import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path

from data_prep.ingestion import scan_cards, load_results_and_merge
from data_prep.readers import load_all_pp_cards
from data_prep.utilities import load_config, setup_logging
from features.feature_engineering import engineer_features

logger = logging.getLogger(__name__)

ID_COLUMNS = ["race_key", "track", "date", "race_no"]

def configure_pipeline():
    setup_logging(level=logging.INFO)
    logger.info("Logging configured.")
    config = load_config()
    return config

def get_paths_and_patterns(config):
    track = config["DEFAULT"].get("track", "SA")
    logger.info(f"Track from config.ini: {track}")
    
    pp_location = config["DEFAULT"].get("pp_location")
    result_location = config["DEFAULT"].get("result_location")
    predict_location = config["DEFAULT"].get("predict_location")
    pp_field_mapping_csv = config["DEFAULT"].get("pp_fields_mapping_location")
    result_field_mapping_csv = config["DEFAULT"].get("result_fields_mapping_location")
    
    pp_track_pattern = f"{track}*.DRF"
    res_track_pattern = f"{track}*.*"
    
    return (track, pp_location, result_location, predict_location,
            pp_field_mapping_csv, result_field_mapping_csv,
            pp_track_pattern, res_track_pattern)

def preprocess_data(X, num_strategy='mean', cat_strategy='most_frequent'):
    X_processed = X.copy()
    bool_cols = X_processed.select_dtypes(include=['bool']).columns
    if bool_cols.any():
        logger.info(f"Converting boolean columns to integers: {list(bool_cols)}")
        X_processed[bool_cols] = X_processed[bool_cols].astype(int)

    numeric_cols = X_processed.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_processed.select_dtypes(exclude=["number"]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy=num_strategy)
        X_num = pd.DataFrame(num_imputer.fit_transform(X_processed[numeric_cols]),
                             columns=numeric_cols, index=X_processed.index)
        X_num = X_num.astype(float)
    else:
        X_num = pd.DataFrame(index=X_processed.index)

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        X_cat = pd.DataFrame(cat_imputer.fit_transform(X_processed[categorical_cols]),
                             columns=categorical_cols, index=X_processed.index)
        for col in X_cat.columns:
            X_cat[col] = pd.Categorical(X_cat[col]).codes
    else:
        X_cat = pd.DataFrame(index=X_processed.index)

    X_processed = pd.concat([X_num, X_cat], axis=1)
    logger.info(f"Preprocessed data shape: {X_processed.shape}")
    return X_processed

def prepare_data_for_modeling_multitarget(df, target_columns):
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_model_data(merged_df, field_map, target_columns=["res_final_time", "res_finish_position", "res_beaten_lengths", "res_odds"]):
    missing_targets = [col for col in target_columns if col not in merged_df.columns]
    if missing_targets:
        raise ValueError(f"Target columns {missing_targets} not in DataFrame. Available columns: {merged_df.columns.tolist()}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling_multitarget(merged_df, target_columns)
    logger.info(f"Data ready. Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def training_phase(pp_location, result_location, pp_field_mapping_csv, result_field_mapping_csv, pp_track_pattern, res_track_pattern):
    target_columns = ["res_final_time", "res_finish_position", "res_beaten_lengths", "res_odds"]

    pp_cards = scan_cards(pp_location, pp_track_pattern)
    pp_all_df, pp_field_map = load_all_pp_cards(pp_cards, pp_field_mapping_csv)
    logger.info(f"Loaded PP data: {pp_all_df.shape}")

    merged_df, _ = load_results_and_merge(pp_all_df, result_location, result_field_mapping_csv, res_track_pattern)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    merged_df = merged_df.dropna(subset=target_columns)
    for col in target_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_model_data(merged_df, pp_field_map, target_columns)

    X_train = X_train.drop(columns=ID_COLUMNS, errors='ignore')
    X_val = X_val.drop(columns=ID_COLUMNS, errors='ignore')
    X_test = X_test.drop(columns=ID_COLUMNS, errors='ignore')

    X_train = pd.DataFrame(preprocess_data(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(preprocess_data(X_val), columns=X_val.columns, index=X_val.index)
    X_test = pd.DataFrame(preprocess_data(X_test), columns=X_test.columns, index=X_test.index)

    training_columns = X_train.columns.tolist()

    from sklearn.preprocessing import StandardScaler
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train_scaled)
    logger.info("Model training completed.")

    return model, merged_df, target_columns, training_columns, pp_field_map, target_scaler

def prediction_phase(predict_location, pp_field_mapping_csv, pp_track_pattern, model, training_columns, target_columns):
    predict_cards = scan_cards(predict_location, pp_track_pattern)
    predict_pp_df, predict_pp_field_map = load_all_pp_cards(predict_cards, pp_field_mapping_csv)
    logger.info(f"Loaded prediction PP data: {predict_pp_df.shape}")

    for col in target_columns:
        if col not in predict_pp_df.columns:
            predict_pp_df[col] = np.nan

    original_df = predict_pp_df.copy()
    X_pred_raw = predict_pp_df.drop(columns=ID_COLUMNS, errors='ignore')

    engineered_predict_df = engineer_features(X_pred_raw, [], predict_pp_field_map)
    logger.info(f"Engineered prediction features shape (before alignment): {engineered_predict_df.shape}")

    if engineered_predict_df.shape[1] == 0:
        logger.warning("No predictive feature columns produced; creating default features using training_columns.")
        engineered_predict_df = pd.DataFrame(0, index=X_pred_raw.index, columns=training_columns)
    else:
        engineered_predict_df = pd.DataFrame(preprocess_data(engineered_predict_df),
                                              columns=engineered_predict_df.columns,
                                              index=engineered_predict_df.index)
    
    engineered_predict_df = engineered_predict_df.reindex(columns=training_columns, fill_value=0)
    logger.info(f"Engineered prediction features shape (after alignment): {engineered_predict_df.shape}")

    predictions = model.predict(engineered_predict_df)
    logger.info("Prediction completed.")

    for idx, target in enumerate(target_columns):
        col_name = "predicted_" + target
        engineered_predict_df[col_name] = predictions[:, idx]

    predicted_fp_col = "predicted_res_finish_position"
    if predicted_fp_col in engineered_predict_df.columns:
        engineered_predict_df[predicted_fp_col] = engineered_predict_df[predicted_fp_col].rank(method="min").astype(int)
        logger.info(f"Post-processed {predicted_fp_col} to integer rankings.")

    output_columns = [
        "track", "date", "race_no", "post_position", "entry",
        "distance_in_yards", "race_type", "todays_trainer", "todays_jockey",
        "horse_name", "predicted_res_final_time", "predicted_res_finish_position",
        "predicted_res_beaten_lengths", "predicted_res_odds"
    ]
    
    final_df = pd.concat([original_df.reset_index(drop=True), engineered_predict_df.reset_index(drop=True)], axis=1)
    final_df = final_df[output_columns]

    output_file = Path("predicted_results.csv")
    final_df.to_csv(output_file, index=False)
    logger.info(f"Predicted results saved to '{output_file}'.")

    return predictions, final_df

def main():
    config = configure_pipeline()
    (track, pp_location, result_location, predict_location,
     pp_field_mapping_csv, result_field_mapping_csv,
     pp_track_pattern, res_track_pattern) = get_paths_and_patterns(config)
    
    model, merged_df, target_columns, training_columns, pp_field_map, target_scaler = training_phase(
        pp_location, result_location, pp_field_mapping_csv, result_field_mapping_csv, pp_track_pattern, res_track_pattern
    )

    predictions, final_df = prediction_phase(
        predict_location, pp_field_mapping_csv, pp_track_pattern, model, training_columns, target_columns
    )
    
    print("Merged training data shape:", merged_df.shape)
    print(predictions)

if __name__ == "__main__":
    main()
''',

    "main/__init__.py": r'''# Empty __init__.py for main package
''',

    "modeling/model_training.py": r'''import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

def remap_labels(y):
    """
    Remap labels. For numeric or multi-dimensional targets (regression),
    return as a NumPy array; for classification, implement mapping as needed.
    """
    y_array = np.array(y)
    if np.issubdtype(y_array.dtype, np.number) or y_array.ndim > 1:
        return y_array, {}
    else:
        mapping = {}  # Implement mapping logic if needed.
        y_mapped = np.array([mapping[label] for label in y])
        return y_mapped, mapping

def lgb_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    num_class = len(np.unique(y_train))
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42,
        'metric': 'multi_logloss'
    }
    
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000)
    feature_importance = model.feature_importance(importance_type='gain')
    
    test_preds = model.predict(X_test)
    n_classes = test_preds.shape[1]
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    test_auc = roc_auc_score(y_test_bin, test_preds, multi_class='ovr')
    logger.info(f'Test AUC: {test_auc:.4f}')
    
    y_pred_class = np.argmax(test_preds, axis=1)
    accuracy = accuracy_score(y_test, y_pred_class)
    logger.info(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred_class)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred_class))
    
    roc_auc = roc_auc_score(y_test_bin, test_preds, multi_class='ovr')
    logger.info(f"Multiclass ROC AUC (OvR): {roc_auc:.4f}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-Validation Accuracy Scores: {scores}")
    logger.info(f"Mean CV Accuracy: {scores.mean()}")

    return model, feature_importance

def random_forest_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    logger.info(f'Validation AUC: {val_auc:.4f}')
    logger.info(f'Test AUC: {test_auc:.4f}')
    return model, feature_importance

def binary_regression_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    logger.info(f"Validation Accuracy: {val_acc:.2f}")

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    logger.info(f"Test Accuracy: {test_acc:.2f}")

    logger.info("Classification Report (Test):\n" + classification_report(y_test, y_test_pred))

    coefs = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)
    logger.info(f"Top 10 features:\n{coefs.head(10)}")

    return model, coefs
''',

    "modeling/__init__.py": r'''# Empty __init__.py for modeling package
'''
}

def update_files(file_map):
    for rel_path, content in file_map.items():
        file_path = Path(rel_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated: {file_path}")

if __name__ == "__main__":
    update_files(files_to_update)
    print("All files updated successfully.")
