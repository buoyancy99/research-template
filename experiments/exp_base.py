from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Dict
import pathlib
import os

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = self._build_algo()

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            print(f"== Executing task: {task} =====================")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.experiment.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.experiment.training.data.num_workers),
                shuffle=self.cfg.experiment.training.data.shuffle,
            )
        else:
            return None

    def _build_validation_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        if validation_dataset:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.experiment.validation.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.experiment.validation.data.num_workers),
                shuffle=self.cfg.experiment.validation.data.shuffle,
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.experiment.test.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.experiment.test.data.num_workers),
                shuffle=self.cfg.experiment.test.data.shuffle,
            )
        else:
            return None

    def train(self) -> None:
        """
        All training happens here
        """
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.experiment.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.experiment.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            max_epochs=self.cfg.experiment.training.max_epochs,
            max_steps=self.cfg.experiment.training.max_steps,
            max_time=self.cfg.experiment.training.max_time,
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            val_check_interval=self.cfg.experiment.validation.check_interval,
            limit_val_batches=self.cfg.experiment.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.experiment.validation.check_epoch,
            accumulate_grad_batches=self.cfg.experiment.training.optim.accumulate_grad_batches,
            precision=self.cfg.experiment.training.precision,
            detect_anomaly=self.cfg.experiment.debug,
        )

        if self.debug:
            self.logger.watch(self.model, log="all")

        trainer.fit(
            self.algo,
            train_dataloaders=self._build_training_loader(),
            val_dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

        trainer = pl.Trainer(
            accelerator="auto", logger=self.logger, devices="auto", callbacks=callbacks, precision=self.cfg.precision
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
        )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if split in ["training", "test", "validation"]:
            return self.compatible_datasets[self.cfg.dataset._name](self.cfg.dataset, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
