"""
utils for submitting to clusters, such as slurm
"""


import getpass
import os
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import click
from colorama import Fore

# This is set below.
REPO_DIR = None


def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"


def submit_slurm_job(
    cfg: DictConfig,
    python_args: str,
    project_root: Path,
):
    log_dir = project_root / "slurm_logs" / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{cfg.name}"
    log_dir.mkdir(exist_ok=True, parents=True)
    (project_root / "slurm_logs" / "latest").unlink(missing_ok=True)
    (project_root / "slurm_logs" / "latest").symlink_to(log_dir, target_is_directory=True)

    slurm_script = cfg.cluster.launch_template.format(
        name=cfg.name,
        log_dir=log_dir,
        email=cfg.cluster.email,
        num_gpus=cfg.cluster.num_gpus,
        num_cpus=cfg.cluster.num_cpus,
        memory=cfg.cluster.memory,
        time=cfg.cluster.time,
        project_root=project_root,
        python_args=python_args,
    )

    slurm_script_path = log_dir / "job.slurm"
    with slurm_script_path.open("w") as f:
        f.write(slurm_script)

    os.system(f"chmod +x {slurm_script_path}")
    os.system(f"sbatch {slurm_script_path}")

    print(f"\n{cyan('script:')} {slurm_script_path}\n {cyan('slurm errors and logs:')} {log_dir}\n")
