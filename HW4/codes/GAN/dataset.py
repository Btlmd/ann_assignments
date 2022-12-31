import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

class Dataset(object):

    def __init__(self, batch_size, path, seed=0):
        if not os.path.isdir(path):
            os.mkdir(path)

        self._training_data = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )

        self._validation_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )

        def worker_init_fn(worker_id):
            import random
            import torch
            import numpy as np
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self._training_loader = DataLoader(
            self._training_data,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True
        )

    @property
    def training_data(self):
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader
