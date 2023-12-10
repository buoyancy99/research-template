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

    """
    This experiment class seems empty because all tasks, [train, validation, test]
    are already implemented in BaseLightningExperiment. If you have a new task,
    you can add the method here and put it in the yaml file in `configurations/experiment`.

    For example, `experiments/example_helloworld.py` has a task called `main`.
    """
