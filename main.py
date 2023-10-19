import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger

from utils.wandb_utils import download_latest_checkpoint, rewrite_checkpoint_for_compatibility
from experiments import build_experiment


def process_checkpointing_cfg(cfg: DictConfig) -> Dict[str, Any]:
    params = {**cfg}
    if "train_time_interval" in params:
        params["train_time_interval"] = timedelta(**params["train_time_interval"])
    return params  # type: ignore


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    # Enforce the correct Python version.
    if sys.version_info.major < 3 or sys.version_info.minor < 9:
        print(
            "Please use Python 3.9+. If on IBM Satori, "
            "install Anaconda3-2022.10-Linux-ppc64le.sh"
        )

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(f"Saving outputs to {output_dir}")
    os.system(f"ln -sfn {output_dir} {output_dir.parents[1]}/latest-run")

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        if "name" not in cfg:
            raise ValueError("must specify a name for the run with command line argument '+name=[name]'")
        # If resuming, merge into the existing run on wandb.
        resume_id = cfg.wandb.get("resume", None)
        name = (
            f"{cfg.name} ({output_dir.parent.name}/{output_dir.name})"
            if resume_id is None
            else None
        )
        logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=name,
            log_model="all",
            save_dir=str(output_dir),
            config=OmegaConf.to_container(cfg),
            id=None if cfg.wandb.get("use_new_id", False) else resume_id,
        )

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            print(f"wandb mode: {wandb.run.settings.mode}")
            wandb.run.log_code(".")
    else:
        logger = None

    # If resuming a run, download the checkpoint.
    resume_id = cfg.wandb.get("resume", None)
    if resume_id is not None:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{resume_id}"
        checkpoint_path = download_latest_checkpoint(
            run_path, Path("outputs/loaded_checkpoints")
        )
        checkpoint_path = rewrite_checkpoint_for_compatibility(checkpoint_path)
    else:
        checkpoint_path = None

    # Set matmul precision (for newer GPUs, e.g., A6000).
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


if __name__ == "__main__":
    run()
