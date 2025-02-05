import logging
from data_prep.preprocess import prepare_data_for_modeling
from data_prep.readers import load_all_pp_cards
from data_prep.ingestion import (
    scan_cards,
    load_results_and_merge
)
from data_prep.readers import load_all_pp_cards
from data_prep.utilities import load_config, setup_logging
from features.ffeature_engineering import engineer_features
# from modeling.model_training import binary_regression_classifier  # Uncomment if needed

logger = logging.getLogger(__name__)

def configure_logging_and_load_config():
    """Configure logging and load the configuration from config.ini."""
    setup_logging(level=logging.INFO)
    logger.info("Logging configured.")
    config = load_config()
    return config

def get_track_and_paths(config):
    """
    Extract track code and file paths (for PP, results, and field mappings)
    from the configuration.
    """
    track = config["DEFAULT"].get("track", "SA")
    logger.info(f"Track from config.ini: {track}")
    
    pp_location = config["DEFAULT"].get("pp_location")  # Directory containing PP files
    result_location = config["DEFAULT"].get("result_location")  # Directory containing result files
    pp_field_mapping_csv = config["DEFAULT"].get("pp_fields_mapping_location")
    result_field_mapping_csv = config["DEFAULT"].get("result_fields_mapping_location")
    
    return track, pp_location, result_location, pp_field_mapping_csv, result_field_mapping_csv

def build_file_patterns(track):
    """
    Construct file patterns based on the track code.
    """
    # For example, if track is "SA", then "SA*.DRF" will match PP files,
    # and "SA*.*" will match result files.
    pp_track_pattern = f"{track}*.DRF"
    res_track_pattern = f"{track}*.*"
    logger.info(f"Using PP file pattern: {pp_track_pattern}")
    logger.info(f"Using result file pattern: {res_track_pattern}")
    return pp_track_pattern, res_track_pattern

def load_past_performance(pp_location, pp_track_pattern, pp_field_mapping_csv):
    """
    Scan for PP files and load Past Performance data.
    Returns the PP DataFrame and its field mapping.
    """
    pp_cards = scan_cards(pp_location, pp_track_pattern)
    pp_all_df, pp_field_map = load_all_pp_cards(pp_cards, pp_field_mapping_csv)
    logger.info(f"Loaded PP data: {pp_all_df.shape}")
    return pp_all_df, pp_field_map

def merge_results_with_pp(pp_all_df, result_location, result_field_mapping_csv, res_track_pattern):
    """
    Load result data from the specified directory and merge it with PP data.
    Returns the merged DataFrame and the result field mapping.
    """
    merged_df, result_field_map = load_results_and_merge(
        pp_all_df, result_location, result_field_mapping_csv, res_track_pattern
    )
    logger.info(f"Merged data shape: {merged_df.shape}")
    logger.info(f"Merged columns: {merged_df.columns.tolist()}")
    return merged_df, result_field_map

def prepare_model_data(merged_df, result_field_map, target_column="res_finish_position"):
    """
    Prepare the merged data for modeling by selecting features and splitting data.
    Raises a ValueError if the target column is not found.
    """
    if target_column not in merged_df.columns:
        raise ValueError(f"Target column '{target_column}' not in DataFrame. Available columns: {merged_df.columns.tolist()}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(
        merged_df, result_field_map, target_col=target_column
    )
    logger.info(f"Data ready. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    # Step 1: Configure logging and load configuration
    config = configure_logging_and_load_config()
    
    # Step 2: Get track code and file/mapping paths from config
    (track, pp_location, result_location, pp_field_mapping_csv, result_field_mapping_csv) = get_track_and_paths(config)
    
    # Step 3: Build file patterns based on the track code
    pp_track_pattern, res_track_pattern = build_file_patterns(track)
    
    # Step 4: Load Past Performance (PP) data
    pp_all_df, pp_field_map = load_past_performance(pp_location, pp_track_pattern, pp_field_mapping_csv)
    
    # Step 5: Load result data and merge with PP data
    merged_df, result_field_map = merge_results_with_pp(pp_all_df, result_location, result_field_mapping_csv, res_track_pattern)
    
    # Step 6: Prepare data for modeling.
    # Adjust the target column name as necessary.
    target_column = "res_finish_position"  # Ensure this exists in merged_df
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_model_data(merged_df, result_field_map, target_column)
    
    # Step 7: Feature Engineering
    # Takes the merged DataFrame and the result field mapping, filters the mapping for predictive features 
    # (where predictive_feature equals "1"), and returns a new DataFrame containing only those columns. 
    
    print("Debug merge shape (inner join):", merged_df.shape)
    print("Sample keys from PP data:", pp_all_df["race_key"].unique()[:5])
    print("Sample keys from Result data:", merged_df["race_key"].unique()[:5])        

    engineered_df = engineer_features(merged_df, result_field_map, pp_field_map)
    print("Engineered features DataFrame:")
    print(engineered_df.shape)

    # Step 7: Print dataset shapes and sample data.
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    print(X_train.head())
    
    # Step 8: (Optional) Train a model.
    # model, feature_importance = binary_regression_classifier(X_train, y_train, X_val, y_val, X_test, y_test)
    logger.info("Model training completed.")

if __name__ == "__main__":
    main()
