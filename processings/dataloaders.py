import numpy as np
import torch

from torch.utils.data import DataLoader

from processings.augmentation import transforms


def data_loaders(Dataset, images_dir, batch_size=32, workers=2, image_size=0.5, aug_scale=0.5, aug_angle=0.5):

    dataset = Dataset(
        images_dir,
        subset="train",
        image_size=image_size,
        transform=transforms(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
    )

    total_count = len(dataset)
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(
        dataset, (train_count, valid_count, test_count)
    )

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )

    loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid, loader_test
