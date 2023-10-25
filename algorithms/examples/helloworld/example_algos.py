from omegaconf import DictConfig
from algorithms.common.base_algo import BaseAlgo


class ExampleAlgo(BaseAlgo):
    def __init__(self, cfg: DictConfig):
        """An algorithm that processes a message by appending prefix & suffix string specified in cfg to it.
        See `configurations/algorithm/example_helloworld.yaml` for the configuration that will be used here.
        """
        super().__init__(cfg)
        self.prefix = cfg.prefix
        self.suffix = cfg.suffix

    def run(self, message: str) -> str:
        """return a message with prefix & suffix."""

        if self.debug:
            print(f"We are debugging! The orignal message is: {message}")

        return f"{self.prefix}{message}{self.suffix}"


class ExampleBackwardAlgo(BaseAlgo):
    def __init__(self, cfg: DictConfig):
        """An algorithm that processes a message by first reversing it and then appending prefix
        & suffix string specified in cfg to it.

        See `configurations/algorithm/example_helloworld.yaml` for the configuration that will be used here.
        """
        super().__init__(cfg)
        self.prefix = cfg.prefix
        self.suffix = cfg.suffix

    def run(self, message: str) -> str:
        """return a message with prefix & suffix."""

        if self.debug:
            print(f"We are debugging! The orignal message is: {message}")

        return f"{self.prefix}{message[::-1]}{self.suffix}"
