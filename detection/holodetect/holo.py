import itertools
from functools import partial
from typing import Counter, List, Tuple
import numpy as np
import torch.nn.functional as F
from detection.base import *
from utils.highway import Highway
from channel.noisy_channel import NCGenerator
from utils.helpers import (
    split_train_test_dls,
    str2regex,
    unzip_and_stack_tensors,
)
from utils.attribute import (
    sym_trigrams,
    sym_value_freq,
    val_trigrams,
    value_freq,
    xngrams,
)
from loguru import logger
from pytorch_lightning import Trainer
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.vectors import FastText


class HoloFeatureExtractor:
    def fit(self, values):
        logger.debug("Values: " + str(values[:10]))
        trigram = [["".join(x) for x in list(xngrams(val, 3))] for val in values]
        ngrams = list(itertools.chain.from_iterable(trigram))
        self.trigram_counter = Counter(ngrams)
        sym_ngrams = [str2regex(x, False) for x in ngrams]
        self.sym_trigram_counter = Counter(sym_ngrams)
        self.val_counter = Counter(values)
        sym_values = [str2regex(x, False) for x in values]
        self.sym_val_counter = Counter(sym_values)
        self.func2counter = {
            val_trigrams: self.trigram_counter,
            sym_trigrams: self.sym_trigram_counter,
            value_freq: self.val_counter,
            sym_value_freq: self.sym_val_counter,
        }

    def transform(self, values):
        feature_lists = []
        for func, counter in self.func2counter.items():
            f = partial(func, counter=counter)
            logger.debug(
                "Negative: %s %s" % (func, list(zip(values[:10], f(values[:10]))))
            )
            logger.debug(
                "Positive: %s %s" % (func, list(zip(values[-10:], f(values[-10:]))))
            )
            feature_lists.append(f(values))

        feature_vecs = list(zip(*feature_lists))
        return np.asarray(feature_vecs)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


class HoloLearnableModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.highway = Highway(
            self.hparams.emb_dim, self.hparams.num_layers, torch.nn.functional.relu
        )
        self.linear = nn.Linear(self.hparams.emb_dim, 1)

    def forward(self, inputs):
        avg_input = torch.mean(inputs, dim=1)
        hw_out = self.highway(avg_input)
        return self.linear(hw_out)


class HoloModel(BaseModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.char_model = HoloLearnableModule(hparams)
        self.word_model = HoloLearnableModule(hparams)
        self.fcs = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hparams.input_dim),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.input_dim, 1),
        )

    def forward(self, word_inputs, char_inputs, other_inputs):
        word_out = self.word_model(word_inputs)
        char_out = self.char_model(char_inputs)
        concat_inputs = torch.cat([char_out], dim=1).float()
        return torch.sigmoid(concat_inputs)

    def training_step(self, batch, batch_idx):
        word_inputs, char_inputs, other_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, char_inputs, other_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        word_inputs, char_inputs, other_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, char_inputs, other_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class HoloDetector(BaseDetector):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.feature_extractor = HoloFeatureExtractor()
        self.tokenizer = get_tokenizer("spacy")
        self.fasttext = FastText()
        self.scaler = MinMaxScaler()
        self.generator = NCGenerator()

    def extract_features(self, data, labels=None):
        if labels:
            features = self.scaler.fit_transform(self.feature_extractor.fit_transform(data))
        else:
            features = self.scaler.transform(self.feature_extractor.transform(data))
        features = torch.tensor(features)
        word_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(self.tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        char_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(list(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        if labels is not None:
            label_data = torch.tensor(labels)
            return word_data, char_data, features, label_data
        return word_data, char_data, features

    def detect_values(self, ec_str_pairs: List[Tuple[str, str]], values: List[str]):
        data, labels = self.generator.fit_transform(ec_str_pairs, values)
        feature_tensors_with_labels = self.extract_features(data, labels)
        dataset = TensorDataset(*feature_tensors_with_labels)
        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.model.batch_size, ratios=[0.7, 0.1], num_workers=self.hparams.model.num_workers
        )
        if len(train_dataloader) > 0:
            self.model = HoloModel(self.hparams.model)
            self.model.train()
            trainer = Trainer(
                gpus=self.hparams.num_gpus,
                accelerator="dp",
                log_every_n_steps=40,
                max_epochs=self.hparams.model.num_epochs,
                auto_lr_find=True
            )
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
        feature_tensors = self.extract_features(values)
        self.model.eval()
        pred = self.model.forward(*feature_tensors)
        return pred.squeeze(1).detach().cpu().numpy()

    def detect(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        result_df = raw_df.copy()
        for column in raw_df.columns:
            values = raw_df[column].values.tolist()
            cleaned_values = cleaned_df[column].values.tolist()
            false_values = []
            for val, cleaned_val in zip(values, cleaned_values):
                if val != cleaned_val:
                    false_values.append((val, cleaned_val))
            if not false_values:
                result_df[column] = pd.Series([True for _ in range(len(raw_df))])
            else:
                outliers = self.detect_values(false_values, values)
                result_df[column] = pd.Series(outliers)
        return result_df
