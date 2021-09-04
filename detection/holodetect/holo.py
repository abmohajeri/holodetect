from typing import List, Tuple
import torch.nn.functional as F
from detection.features import *
from detection.base import *
from utils.highway import Highway
from channel.noisy_channel import NCGenerator
from utils.helpers import *
from pytorch_lightning import Trainer
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.vectors import FastText


class HoloFeatureExtractor:
    def __init__(self):
        self.charword = CharWordExtractor()
        self.statistic = StatsExtractor()

    def extract_charword(self, values):
        return self.charword.fit_transform(values)

    def extract_statistics(self, values):
        return self.statistic.fit_transform(values)


class HoloLearnableModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.highway = Highway(
            self.hparams.emb_dim, self.hparams.num_layers, nn.functional.relu
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
        char_out = self.char_model(char_inputs)
        word_out = self.word_model(word_inputs)
        concat_inputs = torch.cat([char_out, word_out, other_inputs], dim=1).float()
        fcs = self.fcs(concat_inputs)
        return torch.sigmoid(fcs)

    def training_step(self, batch, batch_idx):
        word_inputs, char_inputs, other_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, char_inputs, other_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
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
        self.feature_extractor = HoloFeatureExtractor() # Feature extractor class
        self.tokenizer = get_tokenizer("spacy") # Tokenizer is Spacy
        self.fasttext = FastText()
        self.scaler = MinMaxScaler() # Transform features by scaling each feature to a given range (mainly 0-1)
        self.generator = NCGenerator() # Noisy channel that augments data

    def extract_features(self, data, labels=None):
        # Attribute Level (3-gram, symbolic 3-gram and empirical distribution model (frequency of cell value))
        features = self.scaler.fit_transform(self.feature_extractor.extract_charword(data))
        features = torch.tensor(features)
        # Attribute Level (Word Embedding)
        word_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(self.tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        # Attribute Level (Character Embedding)
        char_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(list(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        # Attribute Level (empirical distribution model (one Hot Column ID; Captures per-column bias))
        if labels is not None:
            label_data = torch.tensor(labels)
            return word_data, char_data, features, label_data
        return word_data, char_data, features

    def detect_values(self, values: List[str], data, labels):
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
                accelerator="dp", # dp is DataParallel (split batch among GPUs of same machine) & ddp is DistributedDataParallel (each gpu on each node trains, and syncs grads)
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

    def detect(self, dataset: pd.DataFrame, training_data: pd.DataFrame):
        result_df = dataset['raw'].copy()
        for column in dataset['raw'].columns:
            values = dataset['raw'][column].values.tolist()
            cleaned_values = dataset['clean'][column].values.tolist()
            training_values = training_data['raw'][column].values.tolist()
            training_cleaned_values = training_data['clean'][column].values.tolist()
            if values == cleaned_values:
                result_df[column] = pd.Series([True for _ in range(len(dataset['raw']))])
            else:
                # Data Augmentation
                data, labels = self.generator.fit_transform(list(zip(training_cleaned_values, training_values)), values)
                outliers = self.detect_values(values, data, labels)
                result_df[column] = pd.Series(outliers)
        return result_df
