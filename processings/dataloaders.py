import numpy as np

from torch.utils.data import DataLoader

from processings.augmentation import transforms


def data_loaders(Dataset, images_dir, batch_size=32, workers=2, image_size=0.5, aug_scale=0.5, aug_angle=0.5):

    dataset_train = Dataset(
        images_dir,
        subset="train",
        image_size=image_size,
        transform=transforms(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
    )

    dataset_valid = Dataset(
        images_dir,
        subset="validation",
        image_size=image_size,
        random_sampling=False,
    )

    dataset_test = Dataset(
        images_dir,
        subset="test",
        image_size=image_size,
        random_sampling=False,
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
