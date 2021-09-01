from sklearn.feature_extraction.text import CountVectorizer
from utils.helpers import str2regex
import numpy as np


class RowBasedValue:
    def __init__(self, value, row):
        self.value = value
        self.row = row


class StatsExtractor:
    def __init__(self):
        self.char_counts = CountVectorizer(analyzer="char", lowercase=False)
        self.word_counts = CountVectorizer(lowercase=False)
        self.regex_counts = CountVectorizer(analyzer="char", lowercase=False)
        self.covalue_counts = CountVectorizer(analyzer=lambda x: x)

    def fit(self, values):
        self.char_counts.fit([val.value for val in values])
        self.word_counts.fit([val.value for val in values])
        self.regex_counts.fit([str2regex(val.value, match_whole_token=False) for val in values])

    def transform(self, values):
        char_features = self.char_counts.transform([val.value for val in values]).todense()
        word_features = self.word_counts.transform([val.value for val in values]).todense()
        regex_features = self.regex_counts.transform([str2regex(val.value, match_whole_token=False) for val in values]).todense()
        co_feature_lists = []
        for i in range(len(values[0].row)):
            covalue_list = []
            for value in values:
                covalue_list.append(f"{value}||{i}||{value.row[i]}")
            co_feature_lists.append(self.covalue_counts.fit_transform(covalue_list))
        return np.concatenate([char_features, word_features, regex_features], axis=1)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

