import rootutils

rootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

from typing import Any, List, Optional

import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from training.src.data.components.cxr_dataset import CXRDataset


class CXRDataModule(LightningDataModule):
    """A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "",
        batch_size: int = 64,
        num_workers: int = 0,
        tile_size: List[int] = [224, 224],
        dirs: List[str] = ["train", "val", "test"],
        pin_memory: bool = False,
        train_augs: bool = False,
        val_augs: bool = False,
        label_selection: List[str] = ["all"],
    ) -> None:
        """Initialize a `CXRDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.dirs = dirs
        self.label_selection = label_selection

        augmentations = [
            T.RandomRotation(15),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]

        base_transforms = [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]

        self.train_transforms = T.Compose(
            augmentations + base_transforms if train_augs else base_transforms
        )

        self.val_transforms = T.Compose(
            augmentations + base_transforms if val_augs else base_transforms
        )

        self.test_transforms = T.Compose(base_transforms)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of CXR classes.
        """
        return len(self.label_selection)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things like random
        split twice! Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once the data is
        prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
                    Defaults to ``None``.
        """
        self.data_train = CXRDataset(
            csv_file=f"{self.data_dir}/train.csv",
            transform=self.train_transforms,
            label_selection=self.label_selection,
        )
        self.data_val = CXRDataset(
            csv_file=f"{self.data_dir}/val.csv",
            transform=self.val_transforms,
            label_selection=self.label_selection,
        )
        self.data_test = CXRDataset(
            csv_file=f"{self.data_dir}/val.csv",
            transform=self.test_transforms,
            label_selection=self.label_selection,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = CXRDataModule()
