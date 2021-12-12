from detection.features.base import BaseExtractor
from sklearn.preprocessing import OneHotEncoder


# Attribute Level (One Hot)
class OneHotExtractor(BaseExtractor):
    def __init__(self):
        self.enc = OneHotEncoder(handle_unknown='ignore')

    def fit(self, values):
        self.enc.fit(values)

    def transform(self, values):
        return self.enc.transform(values)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def n_features(self):
        return self.enc.get_feature_names().shape[0]