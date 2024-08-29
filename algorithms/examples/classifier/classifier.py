"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

import torch
import torch.nn as nn

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models import arch_registry


class Classifier(BasePytorchAlgo):
    """
    BasePytorchAlgo's parent class is a pytorch_lightning module whose parent class is nn.Module.
    Pytorch lightning is basically forward method & training loop in a single class.
    See Pytorch lightning documentation https://lightning.ai/docs/pytorch/stable/ for more details.

    A sample algorithm doing classification for CIFAR-10.
    Adopted from https://github.com/kuangliu/pytorch-cifar
    """

    def __init__(self, cfg):
        """cfg is a DictConfig object defined by `configurations/algorithm/example_classifier.yaml`."""
        self.num_class = cfg.num_class
        self.data_mean = cfg.data_mean
        self.data_std = cfg.data_std
        self.classes = cfg.classes
        super().__init__(cfg)  # superclass saves cfg as self.cfg and calls _build_model

    def _build_model(self):
        """create all pytorch models."""
        self.model = arch_registry[self.cfg.arch](self.num_class, self.cfg.in_channels)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Return the optimizer we want to use."""
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)

    def forward(self, inputs, targets):
        """forward is optional, we have it here to make self.training_step simpler."""
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        accuraccy = predicted.eq(targets).float().mean()
        return loss, accuraccy

    def training_step(self, batch, batch_idx):
        """
        See BasePytorchAlgo class for detailed documentation.
        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.

        Return:
            A loss tensor or loss dictionary.
        """
        inputs, targets = batch
        loss, accuraccy = self.forward(inputs, targets)

        """
        Below we log with pytorch lightning's logger but you can directly import wandb and use 
        `wandb.log` as well. The property self.global_step is the step you can give wandb. 
        """

        if (batch_idx + 1) % 100 == 0:
            self.log_dict({"training/loss": loss, "training/accuracy": accuraccy})

        if (batch_idx + 1) % 1000 == 0:
            self.log_image("training/image", inputs, mean=self.data_mean, std=self.data_std)

        if (batch_idx + 1) % 2000 == 0:
            self.log_video("training/video", inputs, mean=self.data_mean, std=self.data_std)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuraccy = self.forward(*batch)

        self.log_dict({"validation/loss": loss, "validation/accuracy": accuraccy})

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuraccy = self.forward(*batch)

        self.log_dict({"test/loss": loss, "test/accuracy": accuraccy})

        return loss
