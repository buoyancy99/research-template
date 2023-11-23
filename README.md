# Instructions

All experiments can be launched via `python main.py [options]` where sample options for each project is provided below.
For clusters like supercloud and satori, you can run `python scripts/submit_job.py` on login node and input options in
my ui interface. It will automatically generate slurm scripts and run them for you on a compute node.

## Setup

Run `conda create python=3.10 -n [your_env_name]` to create environment.
Run `conda activate [your_env_name]` to activate this environment.
Run `pip install -r requirements.txt` to install all dependencies.

If using VScode, please modify `.vscode/settings.json` so python interpreter is set correctly.

## Run built-in example

Run an example machine-learning experiment with a specified dataset and algorithm:
`python main.py +name=xxxx wandb.mode=online experiment=example_classification dataset=example_cifar10 algorithm=example_classifier`

Run a non-machine-learning example experiment:
`python main.py +name=yyyy wandb.mode=online experiment=hello_world algorithm=hello_algo1`

Run a non-machine-learning example experiment, with different algorithm:
`python main.py +name=zzzz wandb.mode=online experiment=hello_world algorithm=hello_algo2`

## Pass in arguments

We use [hydra](https://hydra.cc) instead of `argparse` to configure arguments at every code level. You can both write a static config in `configuration` folder or, at runtime,
[override part of yur static config](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) with command line arguments. 

For example, arguments `algorithm=example_classifier algorithm.lr=1e-3` will override the `lr` variable set by the yaml file you choose in `configurations/algorithm/example_classifier.yaml`. The argument `wandb.mode=online` will override the `mode` under `wandb` namesspace in the file `configurations/config.yaml`.

All static config and runtime override will be logged to cloud automatically.

## Modify for your own project

First, create a new repository with this template. Make sure the new repository has the name you want to use for wandb
logging.

First [Sign up](https://wandb.ai/site) a wandb account for cloud logging and checkpointing. In command line, run `wandb login` to login.

Add your method and baselines in `algorithms` following the `algorithms/README.md` as well as the example code in
`algorithms/examples/classifier/classifier.py`. For pytorch experiments, write your algorithm as a [pytorch lightning](https://github.com/Lightning-AI/lightning)
`pl.LightningModule` which has extensive
[documentation](https://lightning.ai/docs/pytorch/stable/). For a quick start, read "Define a LightningModule" in this [link](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). Finally, add a yaml config file to `configurations/algorithm` imitating that of `configurations/algorithm/example_classifier.yaml`, for each algorithm you added.

(If doing machine learning) Add your dataset in `datasets` following the `datasets/README.md` as well as the example code in
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

## Debug
We provide a useful debug flag which you can enable by `python main.py debug=True`. This will enable numerical error tracking as well as setting `cfg.debug` to `True` for your algorithms and datasets class. However, this debug flag will make ML code very slow as it automatically tracks all parameter / gradients!

## Hyperparameter Sweep

Launch hyperparameter sweep via: `wandb sweep configurations/sweep/example_sweep.yaml`
Then, launch sweep agents on different servers by running the command printed by the controller (e.g., `wandb agent <agent_id>`).


## Feature Roadmap

| **Features**                  | **This repo** |
| ---------------------------   | ----------------------|
| README level documentation    | :heavy_check_mark: |
| Examples  (ML & Non ML)       | :heavy_check_mark: |
| Cloud checkpoint save / load  | :heavy_check_mark: |
| Cloud logging                 | :heavy_check_mark: |
| Hyper-parameter logging       | :heavy_check_mark: |
| Static yaml configuration     | :heavy_check_mark: |
| Yaml config override by arugment  | :heavy_check_mark: |
| Submit UI for MIT cluster     | :heavy_check_mark: |
| Distributed training          | :heavy_check_mark: |
| Low precision training        | :heavy_check_mark: |
| Distributed hyper-parameter sweep | :heavy_check_mark: |
| Debug mode                    | :heavy_check_mark: |
| PEP 8 Style                   | :heavy_check_mark: |
| Type hints                    | :heavy_check_mark: |
| Wiki style documentation      | :x: |
| Unit test                     | :x: |