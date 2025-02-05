# FastRaceML/features/feature_calculator.py

"""
Implements various domain-specific feature calculations for horse racing.
"""

import pandas as pd

class FeatureCalculator:
    @staticmethod
    def calculate_best_lifetime_speed(data: pd.Series):
        """
        Example: read 'best_bris_speed_life' from the row,
        return 0 if missing or NaN.
        """
        speed = data.get('best_bris_speed_life', 0)
        if pd.isna(speed):
            return 0
        return float(speed)

    @staticmethod
    def calculate_speed_last_race(data: pd.Series):
        speed = data.get('speed_rating_856', 0)
        if pd.isna(speed):
            return 0
        return float(speed)

    # Add more static methods for each feature...
