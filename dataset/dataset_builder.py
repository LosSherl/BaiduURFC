from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
import pathlib
import cv2
import os
from imgaug import augmenters

class MMDataset(Dataset):
    def __init__(self, images_df, root_path, vis_path, augment=True, mode="train"):
        if not isinstance(root_path, pathlib.Path):
            root_path = pathlib.Path(root_path)
        if not isinstance(vis_path, pathlib.Path):
            vis_path = pathlib.Path(vis_path)
        self.images_df = images_df.copy()
        self.augment = augment
        self.vis_path = vis_path
        self.images_df.Id = self.images_df.Id.apply(lambda x : root_path / str(x).zfill(6))
        self.mode = mode
    
    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        x = self.read_image(index)
        visit = self.read_npy(index).transpose(1,2,0)
        if not self.mode == "test":
            y = self.images_df.iloc[index].Target
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augment:
            x = self.augment(x)
        x = transforms.Compose([
            transforms.ToPILImage,
            transforms.ToTensor
        ])(x)
        visit = transforms.ToTensor()(visit)
        return x.float(), visit.float(), y
    
    def read_image(self, index):
        row = self.images_df.iloc(index)
        filename = str(row.Id.absolute())
        image = cv2.imread(filename + ".jpg")
        return image

    def read_npy(self, index):
        row = self.images_df.iloc(index)
        filename = os.path.basename(str(row.Id.absolute()))
        p = os.path.join(self.vis_path.absolute(), filename + ".npy")
        visit = np.load(p)
        return visit

    def augmentor(self, image):
        augmentent = augmenters.Sequential([
            augmenters.Filplr(0.5),
            augmenters.Flipud(0.5),
            augmenters.SomeOf((0,4),[
                augmenters.Affine(rotate=90),
                augmenters.Affine(rotate=180),
                augmenters.Affine(rotate=270),
                augmenters.Affine(shear=(-16,16))
            ]),
            augmenters.OneOf([
                augmenters.GaussianBlur((0,3.0)),
                augmenters.AverageBlur(k=(2,7)),
                augmenters.MedianBlur(k=(3,11))
            ])
        ], random_order=True)

        image_aug = augment.augment_image(image)
        return image_aug