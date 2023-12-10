from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
from types import MethodType
from typing_extensions import override
from functools import wraps
import torch
import wandb
import time
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from wandb_osh.hooks import TriggerWandbSyncHook


class OfflineWandbLogger(WandbLogger):
    """
    Wraps WandbLogger to trigger offline sync hook occasionally.
    This is useful when running on slurm clusters, many of which
    only has internet on login nodes, not compute nodes.
    """

    def __init__(
        self,
        name=None,
        save_dir=".",
        version=None,
        offline=False,
        dir=None,
        id=None,
        anonymous=None,
        project=None,
        log_model=False,
        experiment=None,
        prefix="",
        checkpoint_name=None,
        **kwargs: Any,
    ) -> None:
        communication_dir = Path(".wandb_osh_command_dir")
        communication_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_sync = TriggerWandbSyncHook(communication_dir)
        self.last_sync_time = time.time()
        self.min_sync_interval = 60

        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self._offline = offline

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self.trigger_sync(self._save_dir)
            self.last_sync_time = time.time()
        return out


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    latest.download(root=root)
    return root / "model.ckpt"


REWRITES = []


def rewrite_checkpoint_for_compatibility(path: Optional[Path]) -> Optional[Path]:
    """Rewrite checkpoint to account for old versions. It's fine if this is messy."""
    if path is None:
        return None

    # Load the checkpoint.
    checkpoint = torch.load(path)

    # If necessary, rewrite the checkpoint.
    # This ensures that the current code can load checkpoints from old code versions.
    used_rewrite = False
    state_dict = checkpoint["state_dict"]
    for key, value in list(state_dict.items()):
        for old, new in REWRITES:
            if old in key:
                new_key = key.replace(old, new)
                state_dict[new_key] = value
                del state_dict[key]
                used_rewrite = True

    # If nothing was changed, there's no need to save a new checkpoint.
    if not used_rewrite:
        return path

    # Save the modified checkpoint.
    new_path = path.parent / f"{path.name}.rewrite"
    torch.save(checkpoint, new_path)
    return new_path
