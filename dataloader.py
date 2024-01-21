from nuimages import NuImages
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataclasses import dataclass
import os.path as osp


@dataclass
class SemSegSample:
    image: np.ndarray
    segmentation_mask: np.ndarray


class NuImagesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, nuimages, transform=None):
        self.nuimages = nuimages
        self.transform = transform

    def __len__(self):
        return len(self.nuimages.sample)

    def __getitem__(self, sample_token) -> SemSegSample:
        # if torch.is_tensor(sample_token):
        #     sample_token = sample_token.tolist()

        sample = self.nuimages.get('sample', sample_token)
        key_camera_token = sample['key_camera_token']
        semantic_mask, instance_mask = self.nuimages.get_segmentation(key_camera_token)

        sample_data = self.nuimages.get('sample_data', key_camera_token)
        im_path = osp.join(self.nuimages.dataroot, sample_data['filename'])
        im = Image.open(im_path)

        sem_seg_sample = SemSegSample(image=im, segmentation_mask=semantic_mask)

        if self.transform:
            sem_seg_sample = self.transform(sem_seg_sample)

        return sem_seg_sample
