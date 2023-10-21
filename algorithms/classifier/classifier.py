import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models import arch_registry


class Classifier(BasePytorchAlgo):
    """
    A sample algorithm doing classification for CIFAR-10
    Adopted from https://github.com/kuangliu/pytorch-cifar
    """

    def __init__(self, cfg):
        self.num_class = cfg.num_class
        super().__init__(cfg)  # superclass saves cfg as self.cfg and calls _build_model

    def _build_model(self):
        self.model = arch_registry[self.cfg.arch](
            self.num_class, self.cfg.in_channels)
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
