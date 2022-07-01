import itertools
import random
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Tuple

import regex as re
from loguru import logger

from utils import RowBasedValue
from utils.helpers import xngrams


def wskeep_tokenize(s):
    return re.split(r"([\W\p{P}])", s)  # \W Equivalent to [^A-Za-z0-9_] & \p{P} is "Any punctuation character"


class CharTransform:
    def __init__(self, before_str, after_str):
        self.before_str = before_str
        self.after_str = after_str

    def transform(self, str_value):
        if not self.before_str:
            if not str_value:
                return self.after_str
            else:
                sample_position = random.randrange(len(str_value)) # Random number between 0 and len(str_value)
                return (
                    str_value[:sample_position]
                    + self.after_str
                    + str_value[sample_position:]
                )
        return str_value.replace(self.before_str, self.after_str)

    def __eq__(self, o: "CharTransform"):
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __hash__(self) -> int:
        return hash(f"Rule('{self.before_str}', '{self.after_str}')")

    def validate(self, str_value):
        return self.before_str in str_value

    def __repr__(self) -> str:
        return f"CharTransform('{self.before_str}', '{self.after_str}')"


class WordTransform:
    def __init__(self, before_str, after_str):
        self.before_str = before_str
        self.after_str = after_str

    def transform(self, str_value):
        if not self.before_str:
            if not str_value:
                return self.after_str
            else:
                positions = [(0, 0), (len(str_value), len(str_value))] + [
                    m.span() for m in re.finditer("[\p{P}\p{S}]", str_value) # span() return position of find elements & \p{S} is math symbols, currency signs, dingbats, box-drawing characters, etc.
                ]
                idx = random.randrange(len(positions))
                return (
                    str_value[:positions[idx][0]]
                    + self.after_str
                    + str_value[positions[idx][1]:]
                )
        return str_value.replace(self.before_str, self.after_str)

    def __eq__(self, o: "CharTransform"):
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __hash__(self) -> int:
        return hash(f"Rule('{self.before_str}', '{self.after_str}')")

    def validate(self, str_value):
        return self.before_str in str_value

    def __repr__(self) -> str:
        return f"WordTransform('{self.before_str}', '{self.after_str}')"


class CharNoisyChannel:
    def __init__(self):
        self.rule2prob = None

    def longest_common_substring(self, str1, str2):
        seqMatch = SequenceMatcher(None, str1, str2)
        match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
        if match.size != 0:
            return match.a, match.b, match.size
        else:
            return None

    # Compute the overlap of two strings as 2 ∗ C/S:
    # where C is the number of common characters in the two strings
    # and S is the sum of their lengths
    def similarity(self, str1, str2):
        # collections.Counter count each element of iterable objects
        counter1 = Counter(list(str1))
        counter2 = Counter(list(str2))
        c = counter1 & counter2
        n = sum(c.values())
        try:
            return 2 * n / (len(str1) + len(str2))
        except ZeroDivisionError:
            return 0

    # Algorithm 1: Transformation Learning
    def learn_transformation(self, cleaned_str, error_str):
        if not cleaned_str and not error_str:
            return []

        # valid_trans == Φ
        valid_trans = [CharTransform(cleaned_str, error_str)]

        l = self.longest_common_substring(cleaned_str, error_str) # Output: first_match, second_match, length

        if l is None:
            return valid_trans

        lcv, rcv = cleaned_str[:l[0]], cleaned_str[l[0] + l[2]:]
        lev, rev = error_str[:l[1]], error_str[l[1] + l[2]:]

        if self.similarity(lcv, lev) + self.similarity(rcv, rev) >= self.similarity(lcv, rev) + self.similarity(rcv, lev):
            if lcv or lev:
                valid_trans.append(CharTransform(lcv, lev))
            if rcv or rev:
                valid_trans.append(CharTransform(rcv, rev))
            valid_trans.extend(self.learn_transformation(lcv, lev))
            valid_trans.extend(self.learn_transformation(rcv, rev))
        elif self.similarity(lcv, lev) + self.similarity(rcv, rev) < self.similarity(lcv, rev) + self.similarity(rcv, lev):
            if lcv or rev:
                valid_trans.append(CharTransform(lcv, rev))
            if rcv or lev:
                valid_trans.append(CharTransform(rcv, lev))
            valid_trans.extend(self.learn_transformation(rcv, lev))
            valid_trans.extend(self.learn_transformation(lcv, rev))
        return list(set(valid_trans))

    # Algorithm 2: Empirical Transformation Distribution
    def fit(self, string_pairs):
        transforms = []
        for cleaned_str, error_str in string_pairs:
            transforms.extend(self.learn_transformation(cleaned_str, error_str))
        logger.debug("Transform rules: " + str(transforms))
        counter = Counter(transforms)
        sum_counter = sum(counter.values())
        self.rule2prob = {
            transform: count * 1.0 / sum_counter for transform, count in counter.items()
        }
        return self.rule2prob

    # Algorithm 3: Approximate Noisy Channel Policy
    def get_exceptions(self):
        rule_values = list([x.after_str for x in self.rule2prob.keys()])
        one_grams = Counter(
            itertools.chain.from_iterable(rule_values)
        )
        two_ngrams = Counter(
            itertools.chain.from_iterable(
                ["".join(x) for val in rule_values for x in xngrams(val, 2, add_regex=False)]
            )
        )
        return list(set(list(one_grams.keys()) + list(two_ngrams.keys())))


class WordNoisyChannel(CharNoisyChannel):
    def learn_transformation(self, cleaned_str, error_str):
        if not cleaned_str and not error_str:
            return []

        error_tokens = wskeep_tokenize(error_str)
        cleaned_tokens = wskeep_tokenize(cleaned_str)

        return self.learn_transformation_tokens(cleaned_tokens, error_tokens)

    def learn_transformation_tokens(self, cleaned_tokens, error_tokens):
        if not error_tokens and not cleaned_tokens:
            return []

        valid_trans = [WordTransform("".join(cleaned_tokens), "".join(error_tokens))]

        l = self.longest_common_substring(cleaned_tokens, error_tokens)

        if l is None:
            return valid_trans

        lcv, rcv = cleaned_tokens[:l[0]], cleaned_tokens[l[0] + l[2]:]
        lev, rev = error_tokens[:l[1]], error_tokens[l[1] + l[2]:]

        if self.similarity(lcv, lev) + self.similarity(rcv, rev) >= self.similarity(lcv, rev) + self.similarity(rcv, lev):
            if lcv or lev:
                valid_trans.append(WordTransform("".join(lcv), "".join(lev)))
            if rcv or rev:
                valid_trans.append(WordTransform("".join(rcv), "".join(rev)))
            valid_trans.extend(self.learn_transformation_tokens(lcv, lev))
            valid_trans.extend(self.learn_transformation_tokens(rcv, rev))
        elif self.similarity(lcv, lev) + self.similarity(rcv, rev) < self.similarity(lcv, rev) + self.similarity(rcv, lev):
            if lcv or rev:
                valid_trans.append(WordTransform("".join(lcv), "".join(rev)))
            if rcv or lev:
                valid_trans.append(WordTransform("".join(rcv), "".join(lev)))
            valid_trans.extend(self.learn_transformation_tokens(rcv, lev))
            valid_trans.extend(self.learn_transformation_tokens(lcv, rev))
        return list(set(valid_trans))


class NCGenerator:
    def __init__(self):
        self.char_channel = CharNoisyChannel()
        self.word_channel = WordNoisyChannel()

    def _get_suspicious_chars(self, channel):
        for rule in channel.rule2prob.keys():
            if not rule.before_str:
                yield rule.after_str

    def _get_noisy_chars(self):
        noisy_chars = [rule.after_str for rule in
                       self.char_channel.rule2prob.keys()] if self.char_channel.rule2prob is not None else []
        noisy_word = [rule.after_str for rule in
                      self.word_channel.rule2prob.keys()] if self.word_channel.rule2prob is not None else []
        return noisy_chars + noisy_word

    def _get_noises(self):
        noises = []
        for (key, value) in self.char_channel.rule2prob.items():
            noises.append({'raw': key.after_str, 'clean': key.before_str, 'distribution': value})
        for (key, value) in self.word_channel.rule2prob.items():
            noises.append({'raw': key.after_str, 'clean': key.before_str, 'distribution': value})
        return noises

    def find_noise(self, value):
        noises = []
        for noise in self._get_noises():
            if noise['raw'] in value:
                noises.append(noise)
        return sorted(noises, key=lambda d: d['distribution'], reverse=True)

    def _filter_normal_values(self, channel, values: List[RowBasedValue]):
        suspicious_chars = self._get_suspicious_chars(channel)
        all_remove_values = []
        for c in suspicious_chars:
            removed_values = []
            for val in values:
                if c in val.value:
                    removed_values.append(val)
            if len(removed_values) < len(values) * 0.2:
                all_remove_values.extend(removed_values)
        for value in set(all_remove_values):
            values.remove(value)
        noisy_chars = list(set(self._get_noisy_chars()))
        for c in noisy_chars:
            for val in values:
                if c in val.value:
                    values.remove(val)
                    all_remove_values.append(val)
        logger.debug("Remove_values" + str([x.value for x in all_remove_values]))
        return values

    # Generate Negative Values
    def _generate_transformed_data(self, channel, values: List[RowBasedValue]):
        examples = []
        wait_time = 0
        while len(examples) < len(values) and wait_time <= len(values):
            val = random.choice(values)
            probs = []
            rules = []
            for rule, prob in channel.rule2prob.items():
                if rule.validate(val.value):
                    rules.append(rule)
                    probs.append(prob)
            if probs:
                rule = random.choices(rules, weights=probs, k=1)[0]
                transformed_value = rule.transform(val.value[:])
                tmp = RowBasedValue(transformed_value, val.row, val.column)
                tmp.row[val.column] = transformed_value
                examples.append(tmp)
            else:
                wait_time += 1
        return examples

    def fit_transform_channel(self, channel, ec_pairs: List[Tuple[RowBasedValue, RowBasedValue]], values: List[RowBasedValue]):
        neg_pairs = [(x[0].value, x[1].value) for x in ec_pairs if x[0].value != x[1].value]
        channel.fit(neg_pairs)
        logger.debug("Rule Probabilities: " + str(channel.rule2prob))
        neg_values = [x[1] for x in ec_pairs if x[0].value != x[1].value] + self._generate_transformed_data(channel, values)
        pos_values = [val for val in values if val.value not in list(set([x.value for x in neg_values]))]
        pos_values = self._filter_normal_values(channel, pos_values) + [x[0] for x in ec_pairs if x[0].value == x[1].value]
        # pos_values = [x[0] for x in ec_pairs if x[0].value == x[1].value]
        logger.debug(f"{len(neg_values)} negative values: " + str(list(set([x.value for x in neg_values]))))
        logger.debug(f"{len(pos_values)} positive values: " + str(list(set([x.value for x in pos_values]))))
        data, labels = (
            neg_values + pos_values,
            [0 for _ in range(len(neg_values))] + [1 for _ in range(len(pos_values))],
        )
        return data, labels

    def fit_transform(self, ec_pairs: List[Tuple[RowBasedValue, RowBasedValue]], values: List[RowBasedValue]):
        data1, labels1 = self.fit_transform_channel(self.char_channel, ec_pairs, values)
        data2, labels2 = self.fit_transform_channel(self.word_channel, ec_pairs, values)
        return data1 + data2, labels1 + labels2
