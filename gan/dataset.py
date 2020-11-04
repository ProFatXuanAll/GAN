import torch
import torchvision

import gan.path

class BaseDataset:
    def __init__(self, batch_size: int, is_train: bool, shuffle: bool):
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle

    def get_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def __iter__(self) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_d_in() -> int:
        raise NotImplementedError

    @staticmethod
    def get_d_out() -> int:
        raise NotImplementedError

class MNIST(BaseDataset):
    def get_dataset(self) -> torch.utils.data.Dataset:
        return torchvision.datasets.MNIST(
            gan.path.DATA_PATH,
            train=self.is_train,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )

    def __iter__(self) -> torch.Tensor:
        while True:
            dataset = torch.utils.data.DataLoader(
                dataset=self.get_dataset(),
                batch_size=self.batch_size,
                shuffle=self.shuffle
            )
            for x, y in dataset:
                yield x, y

    @staticmethod
    def get_d_in() -> int:
        return 28 * 28

    @staticmethod
    def get_d_out() -> int:
        return 10


dataset_map = {
    'mnist': MNIST,
}
