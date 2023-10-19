from typing import Optional, Union
from omegaconf import DictConfig
import pathlib
from pytorch_lightning.loggers.wandb import WandbLogger

from .exp_base import BaseExperiment
from .exp_classification import ClassificationExperiment

exp_registry = dict(
    example_classification=ClassificationExperiment,
)


def build_experiment(
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None
) -> BaseExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    return exp_registry[cfg.experiment.name](cfg, logger, ckpt_path)
