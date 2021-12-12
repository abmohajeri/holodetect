from sklearn.feature_extraction.text import CountVectorizer
from detection.features.base import BaseExtractor
import numpy as np


# Tuple Level (Co-occurrence statistics for a cell’s value)
class StatsExtractor(BaseExtractor):
    def __init__(self):
        self.covalue_counter = {}

    def fit(self, values):
        for name in values[0].row.keys():
            if name == values[0].column:
                continue
            covalue_list = []
            for value in values:
                covalue_list.append(f"{' '.join(value.row.values())}||{name}||{value.row[name]}")
            self.covalue_counter[name] = CountVectorizer(
                analyzer=lambda x: [x]
            ).fit(covalue_list)

    def transform(self, values):
        co_feature_lists = []
        for name in values[0].row.keys():
            if name == values[0].column:
                continue
            covalue_list = []
            for value in values:
                covalue_list.append(f"{' '.join(value.row.values())}||{name}||{value.row[name]}")
            co_feature_lists.append(
                self.covalue_counter[name].transform(covalue_list).todense()
            )
        return np.array(co_feature_lists).mean(axis=0)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def n_features(self):
        return len(list(self.covalue_counter.values())[0].get_feature_names())