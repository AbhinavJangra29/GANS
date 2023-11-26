#pil handles image processing tasks
from PIL import Image
#interacting with os to allow file and directory operations
import os
#for creating custom dataset
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    #root directories for zebra and horse
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform
        #Retrieves the list of image filenames for zebras and horses in the specified directories.
        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        #length of maxlen dataset
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) # 1000, 1500
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        #index% ensures index stays inside bounds
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]
        #The result is the full file path to the zebra image.
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img




