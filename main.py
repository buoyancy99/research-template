"""
Main file for the project. This will create and run new experiments and load checkpoints from wandb. 
Borrowed part of the code from David Charatan and wandb.
"""

import os
import sys
import subprocess
import select
import time
import click
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb_osh.syncer import WandbSyncer

from utils.print_utils import cyan
from utils.wandb_utils import download_latest_checkpoint, OfflineWandbLogger
from utils.cluster_utils import submit_slurm_job
from experiments import build_experiment


def process_checkpointing_cfg(cfg: DictConfig) -> Dict[str, Any]:
    params = {**cfg}
    if "train_time_interval" in params:
        params["train_time_interval"] = timedelta(**params["train_time_interval"])
    return params  # type: ignore


def run_local(cfg: DictConfig):
    # Enforce the correct Python version.
    if sys.version_info.major < 3 or sys.version_info.minor < 9:
        print("Please use Python 3.9+. ")

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        cfg.experiment._name = cfg_choice["experiment"]
        cfg.dataset._name = cfg_choice["dataset"]
        cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    print(f"Saving outputs to {output_dir}")
    (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
    (output_dir.parents[1] / "latest-run").symlink_to(output_dir, target_is_directory=True)

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        resume_id = cfg.wandb.get("resume", None)
        name = f"{cfg.name} ({output_dir.parent.name}/{output_dir.name})" if resume_id is None else None

        if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
            logger_cls = OfflineWandbLogger
        else:
            logger_cls = WandbLogger

        logger = logger_cls(
            name=name,
            save_dir=str(output_dir),
            offline=cfg.wandb.mode != "online",
            project=cfg.wandb.project,
            log_model="all",
            config=OmegaConf.to_container(cfg),
            id=None if cfg.wandb.get("use_new_id", False) else resume_id,
        )

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            print(f"wandb mode: {wandb.run.settings.mode}")
            wandb.run.log_code(".")
    else:
        logger = None

    # Resuming a run,
    resume_id = cfg.wandb.get("resume", None)
    if resume_id is not None:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{resume_id}"
        checkpoint_path = Path("outputs/loaded_checkpoints") / run_path / "model.ckpt"
        print(f"Will load checkpoint from {checkpoint_path}")
    else:
        checkpoint_path = None

    # Set matmul precision (for newer GPUs, e.g., A6000).
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


def run_slurm(cfg: DictConfig):
    python_args = " ".join(sys.argv[1:]) + " +_on_compute_node=True"
    project_root = Path.cwd()
    while not (project_root / ".git").exists():
        project_root = project_root.parent
        if project_root == Path("/"):
            raise Exception("Could not find repo directory!")

    slurm_log_dir = submit_slurm_job(
        cfg,
        python_args,
        project_root,
    )

    if "cluster" in cfg and cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
        print("Job submitted to a compute node without internet. This requires manual syncing on login node.")
        osh_command_dir = project_root / ".wandb_osh_command_dir"

        try:
            if click.confirm("Do you want us to run the sync loop for you?", default=True):
                osh_proc = subprocess.Popen(["wandb-osh", "--command-dir", osh_command_dir])
                print(f"Running wandb-osh in background... PID: {osh_proc.pid}")
                print(f"To kill the background sync process, run 'kill {osh_proc.pid}'.")

            print("Once the job gets allocated and starts running, output will be printed below: (Ctrl + C to exit)")
            tail_proc = subprocess.Popen(
                ["tail", "-f", slurm_log_dir / "*.out"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            p = select.poll()
            p.register(tail_proc.stdout)

            while True:
                if p.poll(1):
                    out = tail_proc.stdout.readline().strip().decode("utf-8")
                    if out:
                        print(out)
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"You can manually start a sync loop & get wandb link later by running the following:")
            print(cyan(f"wandb-osh --command-dir {osh_command_dir}"))


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
        with open_dict(cfg):
            if cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
                cfg.wandb.mode = "offline"

    if "name" not in cfg:
        raise ValueError("must specify a name for the run with command line argument '+name=[name]'")

    if not cfg.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'configurations/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group name."
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)

    # If resuming a run and not on a compute node, download the checkpoint.
    resume_id = cfg.wandb.get("resume", None)
    if resume_id and "_on_compute_node" not in cfg:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{resume_id}"
        download_latest_checkpoint(run_path, Path("outputs/loaded_checkpoints"))

    if "cluster" in cfg and not "_on_compute_node" in cfg:
        print(cyan("Slurm detected, submitting to compute node instead of running locally..."))
        run_slurm(cfg)
    else:
        run_local(cfg)


if __name__ == "__main__":
    run()
