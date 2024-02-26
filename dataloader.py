from nuimages import NuImages
from PIL import Image
from torch.utils.data import Dataset, DataLoader, sampler
import numpy as np

import os.path as osp


class NuImagesDataset(Dataset):
    def __init__(self, nuimages, transform=None):
        self.nuimages = nuimages
        self.transform = transform

    def __len__(self):
        return len(self.nuimages.sample)

    def __getitem__(self, sample_token):
        sample = self.nuimages.get('sample', sample_token)
        key_camera_token = sample['key_camera_token']
        semantic_mask, instance_mask = self.nuimages.get_segmentation(key_camera_token)
        # Targets for Torch's cross_entropy function have to be Longs, not Ints.
        semantic_mask = semantic_mask.astype(np.int64)

        sample_data = self.nuimages.get('sample_data', key_camera_token)
        im_path = osp.join(self.nuimages.dataroot, sample_data['filename'])
        im = Image.open(im_path)

        sem_seg_sample = {
            "image": im,
            "segmentation_mask": semantic_mask,
        }

        if self.transform:
            sem_seg_sample = self.transform(sem_seg_sample)

        return sem_seg_sample

    def get_all_keys(self):
        return [sample['token'] for sample in self.nuimages.sample]

def get_mini_dataloader(batch_size: int, transform):
    nuimages = NuImages(dataroot='/data/sets/nuimages', version='v1.0-mini', verbose=False, lazy=True)
    dataset = NuImagesDataset(nuimages, transform=transform)

    all_keys = dataset.get_all_keys()

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(all_keys))

def get_dataloaders(batch_size: int, train_transform, val_transform):
    train_nuimages = NuImages(dataroot='/data/sets/nuimages', version='v1.0-train', verbose=True, lazy=True)
    val_nuimages = NuImages(dataroot='/data/sets/nuimages', version='v1.0-val', verbose=True, lazy=True)

    train_dataset = NuImagesDataset(train_nuimages, transform=train_transform)
    # May want to apply a standardization transform to the val dataset, but doing no transform for now.
    val_dataset = NuImagesDataset(val_nuimages, transform=val_transform)

    train_keys = train_dataset.get_all_keys()
    val_keys = val_dataset.get_all_keys()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_keys))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(val_keys))

    return {"train": train_dataloader, "val": val_dataloader}

def image_transform_to_dict_transform(transform):
    def dict_transform(dic):
        return {k: (transform(v) if k == "image" else v)
                for k, v in dic.items()}
    return dict_transform

