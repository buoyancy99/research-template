# Instructions
All experiments can be launched via `python main.py [options]` where sample options for each project is provided below.
For clusters like supercloud and satori, you can run `python submit_job.py` on login node and input options in
the interface. It will automatically generate slurm scripts and run them for you on a compute node.

## Setup
This repository is tested under Python 3.10. Run `pip install -r requirements.txt` to install all dependencies.

## Example classification
Run an experiment with a specified dataset and algorithm:

`python main.py +name=example_name wandb.mode=online experiment=example_classification dataset=example_cifar10 algorithm=example_classifier`

## Hyperparameter Sweep
Launch hyperparameter sweep via:

`wandb sweep configurations/sweep/sweep.yaml`

Then, launch sweep agents on different servers by running the command printed by the controller (e.g., `wandb agent <agent_id>`).