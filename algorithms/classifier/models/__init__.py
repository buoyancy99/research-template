from .mobilenet_v2 import MobileNetV2
from .resnet import ResNet18, ResNet34

arch_registry = dict(
    mobilenet_v2=MobileNetV2,
    resnet18=ResNet18,
)
