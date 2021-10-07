from abc import abstractmethod
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from loguru import logger


class BaseDetector:
    def __init__(self):
        super().__init__()

    @abstractmethod
    def detect(self, dataset: pd.DataFrame, training_data: pd.DataFrame):
        pass


class BaseModule(LightningModule):
    def training_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x["loss"] for x in outputs], dim=0).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs], dim=0).mean()
        logs = {"train_loss": avg_loss, "train_acc": avg_acc}
        logger.info("training_epoch_end ---> {0}".format({"avg_train_loss": avg_loss, "log": logs, "progress_bar": logs}))

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs], dim=0).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs], dim=0).mean()
        logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        logger.info("validation_epoch_end ---> {0}".format({"avg_val_loss": avg_loss, "log": logs, "progress_bar": logs}))
