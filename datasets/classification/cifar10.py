from omegaconf import DictConfig

import torchvision
import torchvision.transforms as transforms


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, cfg: DictConfig, split="training"):
        self.cfg = cfg
        self.mean = cfg.mean
        self.std = cfg.std
        self.debug = cfg.debug
        if split == "training":
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            train = True
        elif split == "test" or split == "validation":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            train = False
        else:
            raise ValueError(f"split {split} not available for cifar 10s")
        super().__init__(root=cfg.data_dir, train=train, download=True, transform=transform)
