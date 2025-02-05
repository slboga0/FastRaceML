# FastRaceML/features/feature_set.py

"""
Classes for storing and organizing feature values.
"""

class HorseRaceFeature:
    """
    Simple container for a single feature name and value.
    """
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.value = None

    def set_value(self, val):
        self.value = val

    def get_value(self):
        return self.value

class HorseRaceFeatureSet:
    """
    Manages a collection of HorseRaceFeature objects.
    """
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
        return {name: f.get_value() for name, f in self.features.items()}
