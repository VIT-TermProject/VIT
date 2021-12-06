import glob

import torch
import torch.utils.data import Dataset
import cv2
import numpy

from augmentation import addnoise


class PlacesDataset(Dataset):
    def __init__(self, path, normalize=False, augmentation=False):
        self.images = glob.glob(path + '/*')
        self.augmentation = augmentation
        self.normalize = normalize
        self.noise_type = ["gauss","s&p","poisson","speckle"]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        image = cv2.imread(image_path,cv2_IMREAD_COLOR)
        ground_truth = image.copy()

        noise_idx = np.random.randint(0,3,size =1)
        image = addnoise(self.noise_type[noise_idx], image)

        if self.augmentation:
            # Not implemented yet
            # augmentation(image)
        
        if self.normalize:
            image = image / 255.0
            ground_truth = ground_truth / 255.0
            # image = (image/127.5) - 1
            # ground_truth = (ground_truth/127.5) - 1

        image = image.transpose(2,0,1)
        ground_truth = ground_truth.transpose(2,0,1)

        return image, ground_truth


