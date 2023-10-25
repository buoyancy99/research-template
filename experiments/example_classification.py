import torch
import torch.nn as nn
from typing import Optional

import lightning.pytorch as pl
from .exp_base import BaseLightningExperiment

from algorithms.examples.classifier import Classifier
from datasets import CIFAR10Dataset


class ClassificationExperiment(BaseLightningExperiment):
    """
    A classification experiment
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms = dict(
        example_classifier=Classifier,
    )

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets = dict(
        example_cifar10=CIFAR10Dataset,
    )
