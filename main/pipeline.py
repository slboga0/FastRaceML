import logging
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
