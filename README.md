# Instructions

All experiments can be launched via `python main.py [options]` where sample options for each project is provided below.
For clusters like supercloud and satori, you can run `python submit_job.py` on login node and input options in
the interface. It will automatically generate slurm scripts and run them for you on a compute node.

## Setup

Run `conda create python=3.10 -n [your_env_name]` to create environment.
Run `conda activate [your_env_name]` to activate this environment.
Run `pip install -r requirements.txt` to install all dependencies.

## Run built-in example

Run an experiment with a specified dataset and algorithm:

`python main.py +name=example_name wandb.mode=online experiment=example_classification dataset=example_cifar10 algorithm=example_classifier`

## Modify for your own project

First, create a new repository with this template. Make sure the new repository has the name you want to use for wandb
logging.

If using VScode, please modify `.vscode/settings.json` so python interpreter is set correctly.

First [Sign up](https://wandb.ai/site) a wandb account for cloud logging and checkpointing. In command line, run `wandb login` to login.

Add your method and baselines in `algorithms` following the `algorithms/README.md` as well as the example code in
`algorithms/classifier/classifier.py`. For pytorch experiments, write your algorithm as a [pytorch lightning](https://github.com/Lightning-AI/lightning)
`pl.LightningModule` which has extensive
[documentation](https://lightning.ai/docs/pytorch/stable/). For a quick start, read "Define a LightningModule" in this [link](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). Finally, add a yaml config file to `configurations/algorithm` imitating that of `configurations/algorithm/example_classifier.yaml`, for each algorithm you added.

Add your dataset in `datasets` following the `datasets/README.md` as well as the example code in
`datasets/classification.py`. Finally, add a yaml config file to `configurations/dataset` imitating that of
`configurations/dataset/example_cifar10.yaml`, for each dataset you added.

Add your experiment in `experiments` following the `experiments/README.md` as well as the example code in
`experiments/exp_classification.py`. Then register your experiment in `experiments/__init__.py`.
Finally, add a yaml config file to `configurations/experiment` imitating that of
`configurations/experiment/example_classification.yaml`, for each experiment you added.

Modify `configurations/config.yaml` to set `algorithm` to the yaml file you want to use in `configurations/algorithm`;
set `experiment` to the yaml file you want to use in `configurations/experiment`; set `dataset` to the yaml file you
want to use in `configurations/dataset`, or to `null` if no dataset is needed; Notice the fields should not contain the
`.yaml` suffix.

You are all set!

`cd` into your project root. Now you can launch your new experiment with `python main.py +name=example_name wandb.mode=online`. For a debug run, simply remove `wandb.mode=online` to diable cloud logging. You can run baselines or
different datasets by add arguments like `algorithm=[xxx]` or `dataset=[xxx]`. You can also override any `yaml` configurations by following the next section.

## Pass in arguments

We use [hydra](https://hydra.cc) instead of `argparse` to configure arguments at every code level. You can both write a static config in `configuration` folder or, at runtime,
[override part of yur static config](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) without adding any additional code.

All static config and runtime override will be logged to cloud automatically.

## Hyperparameter Sweep

Launch hyperparameter sweep via: `wandb sweep configurations/sweep/example_sweep.yaml`
Then, launch sweep agents on different servers by running the command printed by the controller (e.g., `wandb agent <agent_id>`).
