import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn


from .models import arch_registry


class Classifier(pl.LightningModule):
    """
    A sample algorithm doing classification for CIFAR-10
    Adopted from https://github.com/kuangliu/pytorch-cifar
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug
        self._build_model()

    def _build_model(self):
        self.model = arch_registry[self.cfg.arch](
            self.cfg.num_class, self.cfg.in_channels)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        accuraccy = predicted.eq(targets).float().mean()
        return loss, accuraccy

    def training_step(self, batch, batch_idx):
        loss, accuraccy = self.forward(*batch)

        self.log_dict({
            'training/loss': loss,
            'training/accuracy': accuraccy
        })

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuraccy = self.forward(*batch)

        self.log_dict({
            'validation/loss': loss,
            'validation/accuracy': accuraccy
        })

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuraccy = self.forward(*batch)

        self.log_dict({
            'test/loss': loss,
            'test/accuracy': accuraccy
        })

        return loss
