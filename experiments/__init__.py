from typing import Optional, Union
from omegaconf import DictConfig
import pathlib
from lightning.pytorch.loggers.wandb import WandbLogger

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
    if cfg.experiment.name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment.name} not found in registry {list(exp_registry.keys())}. "
            "Make sure you register it correctly in 'experiments/__init__.py'"
        )

    return exp_registry[cfg.experiment.name](cfg, logger, ckpt_path)
