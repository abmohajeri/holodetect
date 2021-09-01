import numpy as np
import pandas as pd
import regex as re
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from argparse import Namespace
from pathlib import Path
import yaml
import ftfy
from nltk import ngrams


# Read datasets from given path (don't forget that each directory should contain raw & cleaned)
def read_dataset(data_path, data_range=None):
    data_range = [None, None] if data_range is None else data_range
    data_path = Path(data_path)
    raw_path = data_path / "raw"
    cleaned_path = data_path / "cleaned"
    name2raw = {}
    name2cleaned = {}
    name2groundtruth = {}
    if data_range[0] == None:
        data_range[0] = 0
    if data_range[1] == None:
        data_range[1] = len(list(raw_path.iterdir()))
    for file_path in sorted(list(raw_path.iterdir()))[data_range[0]:data_range[1]]:
        name = file_path.name
        name2raw[name] = pd.read_csv(raw_path / name, keep_default_na=False, dtype=str)\
                           .applymap(lambda x: ftfy.fix_text(x))
        name2cleaned[name] = pd.read_csv(cleaned_path / name, keep_default_na=False, dtype=str)\
                               .applymap(lambda x: ftfy.fix_text(x))
        name2groundtruth[name] = name2raw[name] == name2cleaned[name]
    return name2raw, name2cleaned, name2groundtruth


# Load config file
def load_config(config_path):
    config_path = Path(config_path)
    if config_path.is_dir():
        hparams = Namespace(**{})
        for sub_path in config_path.iterdir():
            if sub_path.name.startswith("."):
                continue
            elif sub_path.stem == "config":
                hparams.__dict__.update(load_config(sub_path).__dict__)
            else:
                setattr(hparams, sub_path.stem, load_config(sub_path))
        return hparams
    else:
        param_dict = yaml.load(config_path.open("r"), Loader=yaml.FullLoader)
        hparams = Namespace(**param_dict)
        return hparams


# Compare Function
def not_equal(df1, df2):
    return (df1 != df2) & ~(df1.isnull() & df2.isnull())


# Find two dataframe differences
def diff_dfs(df1, df2, compare_func=not_equal):
    assert (df1.columns == df2.columns).all(), "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        diff_mask = compare_func(df1, df2)
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ["id", "col"]
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        df = pd.DataFrame({"from": changed_from, "to": changed_to}, index=changed.index)
        df["id"] = df.index.get_level_values("id")
        df["col"] = df.index.get_level_values("col")
        return df


# Convert string to regex
def str2regex(x, match_whole_token=True):
    if not match_whole_token:
        try:
            if x is None:
                return ""
            x = re.sub(r"[A-Z]", "A", x)
            x = re.sub(r"[0-9]", "0", x)
            x = re.sub(r"[a-z]", "a", x)
            return x
        except Exception as e:
            print(e, x)
            return x
    try:
        if x is None:
            return ""
        x = re.sub(r"[A-Z]+", "A", x)
        x = re.sub(r"[0-9]+", "0", x)
        x = re.sub(r"[a-z]+", "a", x)
        x = re.sub(r"Aa", "C", x)
        return x
    except Exception as e:
        print(e, x)
        return x


# CollateFn parameter for dataloader: merges a list of samples to form a mini-batch of Tensor
def unzip_and_stack_tensors(tensor):
    transpose_tensors = list(zip(*tensor))
    result = []
    for tensor in transpose_tensors:
        result.append(torch.stack(tensor, dim=0))
    return result


# Split data to train, validation and test
def split_train_test_dls(data, collate_fn, batch_size, ratios=[0.7, 0.2], pin_memory=True, num_workers=16):
    train_length = int(len(data) * ratios[0])
    val_length = int(len(data) * ratios[1])
    train_dataset, val_dataset, test_dataset = random_split(
        data, [train_length, val_length, len(data) - train_length - val_length],
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_dataset,  
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return train_dl, val_dl, test_dl


# Chunk value to ngrams (n is given)
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