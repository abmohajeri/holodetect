import itertools
from functools import partial
from loguru import logger
from typing import Counter
from utils.helpers import *


class CharWordExtractor:
    def fit(self, values): # values is all data from noisy channel (clean + error)
        trigram = [["".join(x) for x in list(xngrams(val, 3))] for val in values]
        ngrams = list(itertools.chain.from_iterable(trigram))
        self.trigram_counter = Counter(ngrams)
        sym_ngrams = [str2regex(x, False) for x in ngrams]
        self.sym_trigram_counter = Counter(sym_ngrams)
        self.val_counter = Counter(values)
        sym_values = [str2regex(x, False) for x in values]
        self.sym_val_counter = Counter(sym_values)
        # Attribute Level
        # func2counter contain 3-gram, symbolic 3-gram and empirical distribution model (frequency of cell value)
        # These functions available in utils.helpers module
        self.func2counter = {
            val_trigrams: self.trigram_counter,
            sym_trigrams: self.sym_trigram_counter,
            value_freq: self.val_counter,
            sym_value_freq: self.sym_val_counter,
        }

    def fit_transform(self, values):
        self.fit(values)
        feature_lists = []
        for func, counter in self.func2counter.items():
            f = partial(func, counter=counter) # counter=counter set default value for func counter argument
            logger.debug(
                "Negative: %s %s" % (func, list(zip(values[:10], f(values[:10]))))
            )
            logger.debug(
                "Positive: %s %s" % (func, list(zip(values[-10:], f(values[-10:]))))
            )
            feature_lists.append(f(values))
        feature_vecs = list(zip(*feature_lists))
        return np.asarray(feature_vecs)
