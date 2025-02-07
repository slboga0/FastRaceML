import pandas as pd

class FeatureCalculator:
    @staticmethod
    def calculate_best_lifetime_speed(data: pd.Series) -> float:
        speed = data.get('best_bris_speed_life', 0)
        return float(speed) if pd.notna(speed) else 0.0

    @staticmethod
    def calculate_speed_last_race(data: pd.Series) -> float:
        speed = data.get('speed_rating_856', 0)
        return float(speed) if pd.notna(speed) else 0.0
