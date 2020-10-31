import os

import torch
from torchvision import transforms
import torchxrayvision as xrv


class TorchConfig:
    if os.getenv("FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                    xrv.datasets.XRayResizer(224)])

torch_config = TorchConfig()
    