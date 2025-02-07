import logging
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
