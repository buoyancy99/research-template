"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC
from typing import Optional, Union, Dict
import os
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf


from omegaconf import DictConfig

from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero

torch.set_float32_matmul_precision("high")


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self, root_cfg: DictConfig, output_dir: Optional[Union[str, Path]], ckpt_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Constructor

        Args:
            root_cfg: configuration file that contains root configuration from project_root/configurations/config.yaml
            output_dir: a directory to save outputs
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.output_dir = Path(output_dir)
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None

        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug

        # some tasks doesn't need logger or algo (e.g. download dataset) so leave for None for now
        self.logger = None
        self.algo = None

    def _build_logger(self):
        self.logger = wandb
        return self.logger

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        self.algo = self.compatible_algorithms[algo_name](self.root_cfg.algorithm)
        return self.algo

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
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

    def _build_logger(self):
        from utils.wandb_utils import OfflineWandbLogger, SpaceEfficientWandbLogger

        output_dir = Path(self.output_dir)
        wandb_cfg = self.root_cfg.wandb

        # Set up logging with wandb.
        if wandb_cfg.mode != "disabled":
            # If resuming, merge into the existing run on wandb.
            resume = self.root_cfg.get("resume", None)
            name = f"{self.root_cfg.name} ({output_dir.parent.name}/{output_dir.name})" if resume is None else None

            if "_on_compute_node" in self.root_cfg and self.root_cfg.cluster.is_compute_node_offline:
                logger_cls = OfflineWandbLogger
            else:
                logger_cls = SpaceEfficientWandbLogger

            self.logger = logger_cls(
                name=name,
                save_dir=str(output_dir),
                offline=wandb_cfg.mode != "online",
                project=wandb_cfg.project,
                log_model="all",
                config=OmegaConf.to_container(self.root_cfg),
                id=resume,
            )

        return self.logger

    def _build_training_loader(self) -> Optional[torch.utils.data.DataLoader]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_validation_loader(self) -> Optional[torch.utils.data.DataLoader]:
        validation_dataset = self._build_dataset("validation")
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        if validation_dataset:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.validation.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[torch.utils.data.DataLoader]:
        test_dataset = self._build_dataset("test")
        shuffle = False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.test.data.shuffle
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.test.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def training(self) -> None:
        """
        All training happens here
        """
        import lightning.pytorch as pl
        from lightning.pytorch.strategies.ddp import DDPStrategy
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

        if not self.algo:
            self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        if not self.logger:
            self._build_logger()

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    self.output_dir / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.fit(
            self.algo,
            train_dataloaders=self._build_training_loader(),
            val_dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        import lightning.pytorch as pl
        from lightning.pytorch.strategies.ddp import DDPStrategy

        if not self.algo:
            self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        if not self.logger:
            self._build_logger()

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        import lightning.pytorch as pl
        from lightning.pytorch.strategies.ddp import DDPStrategy

        if not self.algo:
            self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        if not self.logger:
            self.logger = self._build_logger()

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.test.inference_mode,
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
            return self.compatible_datasets[self.root_cfg.dataset._name](self.root_cfg.dataset, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
