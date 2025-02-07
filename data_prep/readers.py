import logging
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
