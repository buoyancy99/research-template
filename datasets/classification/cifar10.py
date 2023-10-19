from omegaconf import DictConfig

import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, cfg: DictConfig, split='training'):
        self.cfg = cfg
        print(cfg, cfg.keys())
        self.debug = cfg.debug
        if split == 'training':
            transform = transform_train
            train = True
        elif split == 'test' or split == 'validation':
            transform = transform_test
            train = False
        else:
            raise ValueError(f"split {split} not available for cifar 10s")
        super().__init__(root=cfg.data_dir, train=train, download=True, transform=transform)
