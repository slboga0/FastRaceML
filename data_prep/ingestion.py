import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def scan_cards(root_dir, track_pattern="*.DRF"):
    """
    Scans the specified directory (non-recursively) for files matching the track_pattern
    and returns a list of dictionaries representing each found file.

    Each dictionary contains:
        {
            "card": <filename without extension>,
            "filename": <Path object of the file>
        }

    Args:
        root_dir (str or Path): The directory to scan.
        track_pattern (str): The file pattern to match (default: "*.DRF").

    Returns:
        list[dict]: A list of dictionaries for each file matching the pattern.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    # Convert the input to a Path object and remove any extraneous whitespace.
    root_path = Path(str(root_dir).strip())
    if not root_path.exists():
        raise FileNotFoundError(f"Directory '{root_path}' does not exist.")

    # Use glob (non-recursive) to find files in the top-level folder.
    files = list(root_path.glob(track_pattern))
    logger.info(f"Found {len(files)} file(s) matching '{track_pattern}' in '{root_path}'.")

    # Build and return the list of dictionaries.
    card_list = [{"card": file.stem, "filename": file} for file in files]
    return card_list

def load_results_and_merge(pp_all_df, results_dir, result_field_mapping_csv, track_pattern="",
                           result_merge_cols=None):
    """
    Loads result CSV files using a field mapping CSV, constructs a results DataFrame,
    and merges it with the Past Performance (PP) DataFrame on a constructed merge key.
    A new column order is then applied and the merged DataFrame is saved to CSV.
    
    You can optionally supply a dictionary (result_merge_cols) that specifies the 
    result columns to use for the merge key. For example:
    
        {
            "track": "res_track_name",       # if your CSV mapping uses 'res_track_name'
            "date": "res_race_date",           # if your CSV mapping uses 'res_race_date'
            "race_no": "res_race_number",      # if your CSV mapping uses 'res_race_number'
            "name": "res_horse_name"                 # cleaned horse name column
        }
    
    If result_merge_cols is not provided, the following defaults are used:
    
        {
            "track": "res_track_name",
            "date": "res_race_date",
            "race_no": "res_race_number",
            "name": "res_horse_name"
        }
    
    Args:
        pp_all_df (pd.DataFrame): DataFrame containing PP data.
        results_dir (str or Path): Directory containing result CSV files.
        result_field_mapping_csv (str or Path): CSV mapping for result fields.
        track_pattern (str): Optional filename prefix (e.g., "SA") for results.
        result_merge_cols (dict): Optional dictionary specifying which result columns
                                  to use for merging.
        
    Returns:
        tuple: (merged_df, field_map) where:
            - merged_df is the merged DataFrame (with new column order),
            - field_map is the list of dictionaries from the result mapping.
    """
    # Set default merge columns if not provided.
    if result_merge_cols is None:
        result_merge_cols = {
            "track": "res_track_name",
            "date": "res_race_date",
            "race_no": "res_race_number",
            "horse_name": "res_horse_name"
        }
    
    # Read the result field mapping CSV.
    # Skip the first row (which is extra) and specify column names.
    mapping_df = pd.read_csv(result_field_mapping_csv,
                             skiprows=1,
                             header=None,
                             names=["field_position", "field_name", "predictive_feature"])
    # Ensure that "field_position" is an integer.
    mapping_df["field_position"] = mapping_df["field_position"].astype(int)
    
    # Sort by field_position to ensure proper ordering.
    field_map = mapping_df.sort_values(by="field_position").to_dict(orient="records")
    
    # Ensure the results directory exists.
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory '{results_path}' does not exist.")
    
    all_results = []
    logger.info(f"Loading result files from '{results_path}' using pattern '{track_pattern}'")
    
    # Parse each result file (non-recursively) using the field mapping.
    for csv_file in results_path.glob(track_pattern):
        if csv_file.is_file():
            logger.debug(f"Parsing result file: {csv_file}")
            with open(csv_file, "r", encoding="utf-8") as f:
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
    
    # Convert result records into a DataFrame.
    # (Do not prepend "res_" since the mapping already contains the correct names.)
    res_columns = [fm["field_name"] for fm in field_map]
    result_df = pd.DataFrame(all_results, columns=res_columns)
    
       # Save the engineered DataFrame to CSV.
    result_df.to_csv("data\\outputs\\results.csv", index=False)
    logger.info("Engineered features saved to 'engineered_features.csv'.")

    # Construct merge keys.
    # For PP data: expected columns are 'track', 'date', 'race_no', 'horse_name'
    if {"track", "date", "race_no", "horse_name"}.issubset(pp_all_df.columns):
        pp_all_df["race_key"] = (
            pp_all_df["track"].astype(str).str.upper().str.strip() + "_" +
            pp_all_df["date"].astype(str).str.strip() + "_" +
            pp_all_df["race_no"].astype(str).str.strip() + "_" +
            pp_all_df["horse_name"].astype(str).str.upper().str.strip()
        )
    else:
        logger.warning("PP DataFrame missing one or more of: 'track', 'date', 'race_no', 'horse_name'")
    
    # For result data: use the columns specified in result_merge_cols.
    missing_merge_cols = [col for col in result_merge_cols.values() if col not in result_df.columns]
    if missing_merge_cols:
        logger.warning(f"Result DataFrame is missing merge columns: {missing_merge_cols}")
    else:
        result_df["race_key"] = (
            result_df[result_merge_cols["track"]].astype(str).str.upper().str.strip() + "_" +
            result_df[result_merge_cols["date"]].astype(str).str.strip() + "_" +
            result_df[result_merge_cols["race_no"]].astype(str).str.strip() + "_" +
            result_df[result_merge_cols["horse_name"]].astype(str).str.upper().str.strip()
        )


    # Merge PP and result DataFrames on 'race_key'.
    logger.info("Merging PP data with result data...")
    if "race_key" in pp_all_df.columns and "race_key" in result_df.columns:
        merged_df = pd.merge(pp_all_df, result_df, how='left', on='race_key')
        merged_df.drop_duplicates(subset=["race_key"], keep="first", inplace=True)
    else:
        logger.warning("Missing 'race_key' in one or both DataFrames; merge cannot be performed properly.")
        merged_df = pp_all_df.copy()
        
    # Reorder columns based on the merged DataFrame's actual columns.
    # - First, 'race_key'
    # - Then, all columns that do NOT start with "res_" (assumed PP columns)
    # - Finally, all columns that DO start with "res_" (assumed result columns)
    all_cols = list(merged_df.columns)
    pp_cols_order = [col for col in all_cols if not col.startswith("res_") and col != "race_key"]
    res_cols_order = [col for col in all_cols if col.startswith("res_") and col != "race_key"]
    new_column_order = ["race_key"] + pp_cols_order + res_cols_order
    merged_df = merged_df[new_column_order]
    logger.info(f"New merged columns: {merged_df.columns.tolist()}")
    
    # Save merged data to CSV.
    merged_df.to_csv("data\\outputs\\merged_data.csv", index=False)
    logger.info("Merged data saved to 'merged_data.csv'.")
    
    return merged_df, field_map
