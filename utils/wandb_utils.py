from pathlib import Path
from typing import Optional

import torch
import wandb


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_id: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_id)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_id
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
