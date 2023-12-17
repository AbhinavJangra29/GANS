import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class MapDataset(Dataset):
    #the syntax is first define init, len, getitem
    def __init__(self, root_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))#retrieves the directory of input
        self.root_dir = os.path.join(script_dir, root_dir)#sets the root directory with script dir
        self.list_files = os.listdir(self.root_dir)#retrieves a list of file names in the root dir

    def __len__(self):
        return len(self.list_files)#total no. of images in dataset

    def __getitem__(self, index):#gets an item from the dataset at a specified index
        img_file = self.list_files[index]#retirieves file name of image
        img_path = os.path.join(self.root_dir, img_file)#construct a full path to image
        image = np.array(Image.open(img_path))#image converted to numpy array
        input_image = image[:, :600, :]#left half is x
        target_image = image[:, 600:, :]#right half is y
        
        #augmentation and transforations to both x and y
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

if __name__ == "__main__":
    dataset = MapDataset("data/maps/train/")
    loader = DataLoader(dataset, batch_size=5)

    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()
