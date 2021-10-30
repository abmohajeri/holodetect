from sklearn.feature_extraction.text import CountVectorizer
from detection.features.base import BaseExtractor
from utils.helpers import str2regex
import numpy as np


# Tuple Level (Co-occurrence statistics for a cell’s value)
class StatsExtractor(BaseExtractor):
    def __init__(self):
        self.char_counter = CountVectorizer(analyzer="char", lowercase=False)
        self.word_counter = CountVectorizer(lowercase=False)
        self.regex_counter = CountVectorizer(analyzer="char", lowercase=False)
        self.covalue_counter = {}

    def fit(self, values):
        self.char_counter.fit([val.value for val in values])
        self.word_counter.fit([val.value for val in values])
        self.regex_counter.fit(
            [str2regex(val.value, match_whole_token=False) for val in values]
        )
        for name in values[0].row.keys():
            if name == values[0].column:
                continue
            covalue_list = []
            for value in values:
                covalue_list.append(f"{value.row.values()}||{name}||{value.row[name]}")
            self.covalue_counter[name] = CountVectorizer(
                analyzer=lambda x: [x]
            ).fit(covalue_list)

    def transform(self, values):
        char_features = self.char_counter.transform(
            [val.value for val in values]
        ).todense()
        word_features = self.word_counter.transform(
            [val.value for val in values]
        ).todense()
        regex_features = self.regex_counter.transform(
            [str2regex(val.value, match_whole_token=False) for val in values]
        ).todense()

        co_feature_lists = []
        for name in values[0].row.keys():
            if name == values[0].column:
                continue
            covalue_list = []
            for value in values:
                covalue_list.append(f"{value.row.values()}||{name}||{value.row[name]}")
            co_feature_lists.append(
                self.covalue_counter[name].transform(covalue_list).todense()
            )
        return np.concatenate([char_features, word_features, regex_features], axis=1), np.array(co_feature_lists).mean(axis=0)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def n_features(self):
        return sum(
            [
                len(x.get_feature_names())
                for x in [self.char_counter, self.word_counter, self.regex_counter]
            ]
        ) + len(list(self.covalue_counter.values())[0].get_feature_names())