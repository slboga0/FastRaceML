# FastRaceML/data_prep/readers.py

"""
Contains functions to read past performance cards and results from files.
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def load_single_card(pp_path, field_map):
    """
    Read a single Past Performance file and return a list of dicts.
    Each dict maps field_map's 'field_name' to the corresponding value in each line.
    """
    all_pp_list = []
    try:
        with open(pp_path, "r") as f:
            for counter, line in enumerate(f, start=1):
                line_split = line.strip().split(',')
                record = {}
                # field_map is a list of dicts: [{"field_name": "...", "predictive_feature": 1}, ...]
                for idx, fm in enumerate(field_map):
                    field_name = fm["field_name"]
                    record[field_name] = line_split[idx].strip('"') if idx < len(line_split) else ""
                all_pp_list.append(record)
    except FileNotFoundError:
        logger.error(f"File not found: {pp_path}")
    except Exception as e:
        logger.error(f"Error reading {pp_path}: {e}")
    return all_pp_list

def load_all_pp_cards(pp_cards, field_mapping_csv):
    """
    Iterates over pp_cards = [
       {"card": <file_stem>, "filename": Path(...)},
       ...
    ],
    loads each into a list of dicts, concatenates into one DataFrame.
    Returns:
        pp_all_df (pd.DataFrame)
        field_map (list[dict]) 
    """
    # Load field mappings (CSV must contain columns 'field_name' and 'predictive_feature')
    mapping_df = pd.read_csv(field_mapping_csv)
    field_map = mapping_df[['field_name', 'predictive_feature']].to_dict(orient='records')

    all_pps = []
    for card in pp_cards:
        filename = card.get("filename")
        if not filename:
            logger.warning(f"Skipping card with no filename: {card}")
            continue
        single_card_list = load_single_card(filename, field_map)
        all_pps.extend(single_card_list)

    pp_all_df = pd.DataFrame(all_pps)
    return pp_all_df, field_map

def load_all_result(results_dir, track_pattern=""):
    """
    Scans a directory for CSV files matching the given track pattern, 
    reads each into a list of dictionaries, and returns a tuple 
    of (records, columns).

    Args:
        results_dir (str or Path): The root directory containing your result files.
        track_pattern (str, optional): A pattern prefix for filenames 
                                       (e.g., 'SA', 'AQU', etc.). Defaults to "".

    Returns:
        tuple: (all_results, columns)
            - all_results (list[dict]): Each dict represents one row from any matching CSV file.
            - columns (list[str]): A list of expected column names (placeholder, modify as needed).

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory '{results_path}' does not exist.")

    # Adjust columns to your actual CSV structure
    columns = ["track", "date", "race_no", "entry", "finish", "other_cols"]

    all_results = []

    logger.info(f"Searching for CSV files in '{results_path}' with pattern '{track_pattern}'")

    for csv_file in results_path.rglob(track_pattern):
        if csv_file.is_file():
            logger.debug(f"Reading file: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                # Convert each row to a dict and append to all_results
                for _, row in df.iterrows():
                    record = row.to_dict()
                    all_results.append(record)
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
        else:
            logger.debug(f"Skipping non-file path: {csv_file}")

    logger.info(f"Total records loaded: {len(all_results)} from pattern '{track_pattern}'")

    return all_results, columns
