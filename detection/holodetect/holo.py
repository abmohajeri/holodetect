from typing import List
import torch.nn.functional as F
from detection.base import *
from detection.features import StatsExtractor, CharWordExtractor, AlphaFeatureExtractor, OneHotExtractor
from utils.dcparser import Parser, ViolationDetector
from utils.highway import Highway
from utils.row import RowBasedValue
from channel.noisy_channel import NCGenerator
from utils.helpers import *
from pytorch_lightning import Trainer
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset
from sklearn.preprocessing import MinMaxScaler
torch.cuda.empty_cache()


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
        self.learnable_module = HoloLearnableModule(hparams)
        self.fcs = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hparams.input_dim),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.input_dim, 1),
        )

    def forward(self, char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, val_stats, dc_features, onehot_features):
        char_out = self.learnable_module(char_inputs)
        word_out = self.learnable_module(word_inputs)
        tuple_emb_out = self.learnable_module(tuple_embedding)
        concat_inputs = torch.cat([char_out, word_out, charword, tuple_emb_out, co_val_stats, val_stats, dc_features, onehot_features], dim=1).float()
        # print('Debug of Input input_dim')
        # print(self.hparams.input_dim)
        # print(concat_inputs.shape)
        # print('End Debug of Input input_dim')
        fcs = self.fcs(concat_inputs)
        return torch.sigmoid(fcs)

    def training_step(self, batch, batch_idx):
        char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, val_stats, dc_features, onehot_features, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, val_stats, dc_features, onehot_features)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, val_stats, dc_features, onehot_features, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, val_stats, dc_features, onehot_features)
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
        self.generator = NCGenerator() # Noisy channel that augments data
        self.scaler = MinMaxScaler()  # Transform features by scaling each feature to a given range (mainly 0-1)
        self.alpha_feature_extractor = AlphaFeatureExtractor()
        self.format_feature_extractor = CharWordExtractor() # Attribute Level (3-gram, symbolic 3-gram and empirical distribution model)
        self.stats_feature_extractor = StatsExtractor()
        self.onehot_feature_extractor = OneHotExtractor()

    def extract_features(self, data, labels=None, constraints=None):
        column = data[0].column
        df = pd.DataFrame([x.row.values() for x in data], columns=data[0].row.keys())
        data_values = [x.value for x in data]

        char_data, word_data = self.alpha_feature_extractor.extract_embedding(data_values)

        tuple_embedding = self.alpha_feature_extractor.extract_coval_embedding(data)

        # neighbor_embedding = self.alpha_feature_extractor.extract_neighbor_embedding(data)

        df['u_id'] = df.index
        if constraints is not None:
            dc_errors = ViolationDetector(df, constraints).detect_noisy_cells()
            dc_errors = dc_errors[[True if column == x['column'] else False for i, x in dc_errors.iterrows()]]
            dc_errors_count = dc_errors.groupby('u_id').size().reset_index().rename(columns={0: 'count'})
            dc_errors_count = df[['u_id']].astype(int).merge(dc_errors_count, how='left').fillna(0)
        else:
            dc_errors_count = df[['u_id']].astype(int)
            dc_errors_count['count'] = 0

        if labels is not None:
            onehot_features = torch.tensor(self.onehot_feature_extractor.fit_transform(data).todense())
            error_labels_index = [i for i, x in enumerate(labels) if x == 0]
            dc_features = torch.tensor([[row['count']] if index in error_labels_index else [0] for index, row in
                                        dc_errors_count.iterrows()])
            charword_features = torch.tensor(self.scaler.fit_transform(self.format_feature_extractor.fit_transform(data_values)))
            val_stats, co_val_stats = self.stats_feature_extractor.fit_transform(data)
            val_stats = torch.tensor(val_stats)
            co_val_stats = torch.tensor(co_val_stats)
            self.hparams.model.input_dim = self.stats_feature_extractor.n_features() + self.onehot_feature_extractor.n_features() + 8
        else:
            onehot_features = torch.tensor(self.onehot_feature_extractor.transform(data).todense())
            dc_features = torch.tensor([[row['count']] for index, row in dc_errors_count.iterrows()])
            charword_features = torch.tensor(self.scaler.transform(self.format_feature_extractor.transform(data_values)))
            val_stats, co_val_stats = self.stats_feature_extractor.transform(data)
            val_stats = torch.tensor(val_stats)
            co_val_stats = torch.tensor(co_val_stats)

        if labels is not None:
            return char_data, word_data, charword_features, tuple_embedding, co_val_stats, val_stats, dc_features, onehot_features, torch.tensor(labels)

        return char_data, word_data, charword_features, tuple_embedding, co_val_stats, val_stats, dc_features, onehot_features

    def detect_values(self, row_values, data, labels, constraints):
        feature_tensors_with_labels = self.extract_features(data, labels, constraints)
        dataset = TensorDataset(*feature_tensors_with_labels)
        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.model.batch_size, ratios=[0.7, 0.1], num_workers=self.hparams.model.num_workers
        )
        if len(train_dataloader) > 0:
            self.model = HoloModel(self.hparams.model)
            self.model.train()
            trainer = Trainer(
                gpus=self.hparams.model.num_gpus,
                accelerator="dp", # dp is DataParallel (split batch among GPUs of same machine) & ddp is DistributedDataParallel (each gpu on each node trains, and syncs grads)
                log_every_n_steps=30,
                max_epochs=self.hparams.model.num_epochs,
                auto_lr_find=True
            )
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
        feature_tensors = self.extract_features(row_values, constraints=constraints)
        self.model.eval()
        pred = self.model.forward(*feature_tensors)
        return pred.squeeze(1).detach().cpu().numpy()

    def detect(self, dataset: pd.DataFrame, training_data: pd.DataFrame):
        result_df = dataset['raw'].copy()
        for column in dataset['raw'].columns:
            cleaned_values = dataset['clean'][column].values.tolist()
            values = dataset['raw'][column].values.tolist()
            if values == cleaned_values:
                result_df[column] = pd.Series([True for _ in range(len(dataset['raw']))])
            else:
                row_values = [
                    RowBasedValue(value, row_dict, column)
                    for value, row_dict in zip(values, dataset['raw'].to_dict("records"))
                ]
                training_cleaned_values = [
                    RowBasedValue(value, row_dict, column)
                    for value, row_dict in zip(training_data['clean'][column].values.tolist(), training_data['clean'].to_dict("records"))
                ]
                training_values = [
                    RowBasedValue(value, row_dict, column)
                    for value, row_dict in zip(training_data['raw'][column].values.tolist(), training_data['raw'].to_dict("records"))
                ]
                # Data Augmentation
                data, labels = self.generator.fit_transform(list(zip(training_cleaned_values, training_values)), row_values)
                outliers = self.detect_values(row_values, data, labels, training_data['constraints'])
                result_df[column] = pd.Series(outliers)
        return result_df
