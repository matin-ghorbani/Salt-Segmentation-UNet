import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import cv2 as cv


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transforms: Compose) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx) -> tuple:
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return img, mask
