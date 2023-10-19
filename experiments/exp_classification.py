import torch
import torch.nn as nn
from typing import Optional

import lightning.pytorch as pl
from .exp_base import BaseLightningExperiment

from omegaconf import DictConfig
from algorithms.classifier import Classifier
from datasets import CIFAR10Dataset


class ClassificationExperiment(BaseLightningExperiment):
    """
    A classification experiment
    """
    compatible_algorithms = dict(
        example_classifier=Classifier
    )

    compatible_datasets = dict(
        example_cifar10=CIFAR10Dataset
    )

    def _build_model(self) -> pl.LightningModule:
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        return self.compatible_algorithms[self.cfg.algorithm.name](self.cfg.algorithm)
