import copy

import torch.nn.functional as F
from captum.attr import IntegratedGradients, LRP, FeatureAblation
from captum.attr._utils.lrp_rules import EpsilonRule
from pytorch_lightning import Trainer
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset

from channel.noisy_channel import NCGenerator
from detection.base import *
from detection.features import StatsExtractor, CharWordExtractor, AlphaFeatureExtractor, OneHotExtractor
from utils import LabelEncoderRobust
from utils.dcparser import ViolationDetector
from utils.helpers import *
from utils.highway import Highway
from utils.row import RowBasedValue

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

    def concat_input(self, char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding):
        char_inputs = self.learnable_module(char_inputs)
        word_inputs = self.learnable_module(word_inputs)
        tuple_embedding = self.learnable_module(tuple_embedding)
        concat_inputs = torch.cat([char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding], dim=1).float()
        # print('Debug of Input input_dim')
        # print(self.hparams.input_dim)
        # print(concat_inputs.shape)
        # print('End Debug of Input input_dim')
        return concat_inputs

    def forward(self, concat_inputs):
        fcs = self.fcs(concat_inputs)
        return torch.sigmoid(fcs)

    def training_step(self, batch, batch_idx):
        char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding, labels = batch
        labels = labels.view(-1, 1)
        concat_inputs = self.concat_input(char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding)
        probs = self.forward(concat_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding, labels = batch
        labels = labels.view(-1, 1)
        concat_inputs = self.concat_input(char_inputs, word_inputs, charword, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding)
        probs = self.forward(concat_inputs)
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
        self.generator = NCGenerator()  # Noisy channel that augments data
        self.scaler = MinMaxScaler()  # Transform features by scaling each feature to a given range (mainly 0-1)
        self.alpha_feature_extractor = AlphaFeatureExtractor()
        self.format_feature_extractor = CharWordExtractor()  # Attribute Level (3-gram, symbolic 3-gram and empirical distribution model)
        self.stats_feature_extractor = StatsExtractor()
        self.label_encoder = LabelEncoderRobust()  # Use for one hot encoding
        self.onehot_feature_extractor = OneHotExtractor()

    def extract_features(self, data, labels=None, constraints=None):
        column = data[0].column
        df = pd.DataFrame([x.row.values() for x in data], columns=data[0].row.keys())
        df['u_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1  # For DCs
        data_values = [x.value for x in data]

        char_data, word_data = self.alpha_feature_extractor.extract_embedding(data_values)

        tuple_embedding = self.alpha_feature_extractor.extract_tuple_embedding(data)

        neighbor_embedding = self.alpha_feature_extractor.extract_neighbor_embedding(data_values)

        if constraints is not None:
            dc_errors = ViolationDetector(df, constraints).detect_noisy_cells()
            dc_errors = dc_errors[[True if column == x['column'] else False for i, x in dc_errors.iterrows()]]
            if len(dc_errors):
                dc_errors_count = dc_errors.groupby('u_id').size().reset_index().rename(columns={0: 'count'})
                dc_errors_count = df[['u_id']].astype(int).merge(dc_errors_count, how='left').fillna(0)
            else:
                dc_errors_count = df[['u_id']].astype(int)
                dc_errors_count['count'] = 0
        else:
            dc_errors_count = df[['u_id']].astype(int)
            dc_errors_count['count'] = 0

        if labels is not None:
            dc_features = torch.tensor([[row['count']] for index, row in dc_errors_count.iterrows()])

            charword_features = torch.tensor(self.scaler.fit_transform(self.format_feature_extractor.fit_transform(data_values)))

            self.label_encoder.fit(data_values)
            targets = self.label_encoder.transform(data_values).reshape(-1, 1)
            onehot_features = torch.tensor(self.onehot_feature_extractor.fit_transform(targets).todense())

            co_val_stats = torch.tensor(self.stats_feature_extractor.fit_transform(data))

            self.hparams.input_dim = co_val_stats.shape[1] + onehot_features.shape[1] + 9
        else:
            dc_features = torch.tensor([[row['count']] for index, row in dc_errors_count.iterrows()])

            charword_features = torch.tensor(self.scaler.transform(self.format_feature_extractor.transform(data_values)))

            targets = self.label_encoder.transform(data_values).reshape(-1, 1)
            onehot_features = torch.tensor(self.onehot_feature_extractor.transform(targets).todense())

            co_val_stats = torch.tensor(self.stats_feature_extractor.transform(data))

        if labels is not None:
            return char_data, word_data, charword_features, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding, torch.tensor(labels)

        return char_data, word_data, charword_features, tuple_embedding, co_val_stats, dc_features, onehot_features, neighbor_embedding

    def detect_values(self, row_values, data, labels, constraints):
        feature_tensors_with_labels = self.extract_features(data, labels, constraints)
        dataset = TensorDataset(*feature_tensors_with_labels)
        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.batch_size, ratios=[0.7, 0.1], num_workers=self.hparams.num_workers
        )
        if len(train_dataloader) > 0:
            self.model = HoloModel(self.hparams)
            self.model.train()
            # dp is DataParallel (split batch among GPUs of same machine)
            # ddp is DistributedDataParallel (each gpu on each node trains, and syncs grads)
            trainer = Trainer(
                gpus=self.hparams.num_gpus,
                accelerator="dp",
                logger=False,
                max_epochs=self.hparams.num_epochs,
                auto_lr_find=True
            )
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )

        feature_tensors = self.extract_features(row_values, constraints=constraints)
        individuals = self.model.concat_input(*feature_tensors)
        self.model.eval()
        pred = self.model.forward(individuals)
        predictions = pred.squeeze(1).detach().cpu().numpy()

        # Let's start with the interpretation
        y_hat = [1 if x >= 0.5 else 0 for x in predictions]

        # Explanations
        column = data[0].column

        # IG
        to_be_df = []
        ig_xai = IntegratedGradients(self.model.fcs)
        ig_importances = ig_xai.attribute(individuals).detach().cpu().numpy()

        # LRP
        self.model.fcs[2].rule = EpsilonRule()
        lrp_xai = LRP(self.model.fcs)
        lrp_importances = lrp_xai.attribute(individuals).detach().cpu().numpy()

        # Ablation
        feature_mask = \
            [0, 1, 2, 2, 2, 2, 3] + \
            [4 for x in range(len(feature_tensors_with_labels[4][0]))] + \
            [5] + \
            [6 for x in range(len(feature_tensors_with_labels[6][0]))] + \
            [7]
        ablation_xai = FeatureAblation(self.model.fcs)
        ablation_importances = ablation_xai.attribute(individuals,
                                                      feature_mask=torch.tensor(feature_mask)).detach().cpu().numpy()

        for i in range(len(ig_importances)):
            ig_importance = ig_importances[i]
            lrp_importance = lrp_importances[i]
            ablation_importance = ablation_importances[i]
            for type, values in [
                ("ig", ig_importance), ("lrp", lrp_importance), ("ablation", ablation_importance)
            ]:
                for j, name in enumerate(["char_embedding",
                                          "word_embedding",
                                          "frequencies",
                                          "tuple_embedding",
                                          "co_val_stats",
                                          "dc_features",
                                          "onehot_features",
                                          "neighbor_embedding"]):
                    to_be_df.append({
                        "type": type,
                        "feature": name,
                        "value": values[j],
                    })
        df = pd.DataFrame(to_be_df)
        df.to_csv("xai/{}_xai.csv".format(column))

        df = pd.DataFrame([x.row.values() for x in row_values], columns=data[0].row.keys())
        df['u_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1  # For DCs
        dc_errors = ViolationDetector(df, constraints).detect_noisy_cells()
        dc_errors = dc_errors[[True if column == x['column'] else False for i, x in dc_errors.iterrows()]]
        if len(dc_errors):
            dc_errors_count = dc_errors.groupby('u_id').agg({'attribute': '-'.join}).reset_index()
            dc_errors_count = df[['u_id']].astype(int).merge(dc_errors_count, how='left').fillna(0)
            dc_errors_count[['attribute']].to_csv("xai/{}_dc.csv".format(column), index=False)
        else:
            dc_errors_count = df[['u_id']].astype(int)
            dc_errors_count['attribute'] = 0
            dc_errors_count[['attribute']].to_csv("xai/{}_dc.csv".format(column), index=False)

        noises = []
        for value in row_values:
            noises.append({value.value: self.generator.find_noise(value.value)})
        np.save("xai/{}_noises.npy".format(column), noises)

        return predictions

    def detect(self, dataset: pd.DataFrame, training_data: pd.DataFrame):
        result_df = dataset['raw'].copy()
        for column in dataset['raw'].columns:
            cleaned_values = dataset['clean'][column].values.tolist()
            values = dataset['raw'][column].values.tolist()
            if values == cleaned_values:
                result_df[column] = pd.Series([1.0 for _ in range(len(dataset['raw']))])
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
                temp_training_row_values = copy.deepcopy(row_values)
                data, labels = self.generator.fit_transform(list(zip(training_cleaned_values, training_values)), temp_training_row_values)
                outliers = self.detect_values(row_values, data, labels, training_data['constraints'])
                result_df[column] = pd.Series(outliers)
        return result_df
