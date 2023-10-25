The `datasets` folder is used to contain dataset code or environment code.
Create a folder to create your own pytorch dataset definition. Then, update the `__init__.py`
at every level to register all datasets.

For raw data, please use the `data` folder not `datasets`.

Each dataset class takes in a DictConfig file `cfg` in its `__init__`, which allows you to pass in arguments via configuration file in `configurations/dataset` or [command line override](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).
