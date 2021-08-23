from nltk import ngrams
from utils.helpers import str2regex


def xngrams(value, n, add_regex=True):
    if add_regex:
        value = "^" + value + "$"
    if len(value) >= n:
        return ["".join(ngram) for ngram in ngrams(list(value), n)]
    else:
        return [value]


def val_trigrams(values, counter):
    sum_count = sum(counter.values())
    val_trigrams = [["".join(x) for x in list(xngrams(val, 3))] for val in values]
    res = [
        (min([counter[gram] for gram in trigram]) / sum_count if trigram else 0) * 1.0
        for trigram in val_trigrams
    ]
    return res


def sym_trigrams(values, counter):
    patterns = list(map(lambda x: str2regex(x, False), values))
    return val_trigrams(patterns, counter)


def value_freq(values, counter):
    sum_couter = sum(counter.values())
    return [counter[value] * 1.0 / sum_couter for value in values]


def sym_value_freq(values, counter):
    patterns = list(map(lambda x: str2regex(x, True), values))
    return value_freq(patterns, counter)
