import glob

import torch
from torch.utils.data import Dataset
import cv2
import numpy

#from augmentation import addnoise


class PlacesDataset(Dataset):
    def __init__(self, path, data_type = 'train', normalize=False, augmentation=False):
        self.data_type = data_type
        self.images = glob.glob(path + '/blur/*')
        self.images += glob.glob(path + '/gaussian/*')
        self.images += glob.glob(path + '/salt_pepper/*')
        self.augmentation = augmentation
        self.normalize = normalize
        #self.noise_type = ["gauss","s&p","poisson","speckle"]
        self.backup = self.images[0]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        name = image_path.rsplit('/',1)[1]
        gt_split = name.replace('-','/').rsplit('_',1)
        gt_split = gt_split[0] + '/' + gt_split[1]
        
        if self.data_type == 'train':
            gt_path = "../DATA/GT/train/data_256/%s/%s" % (name[0],gt_split)
        else:
            gt_path = "../DATA/GT/val/val_256/%s" % name
            
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        ground_truth = cv2.imread(gt_path,cv2.IMREAD_COLOR)
        ground_truth = cv2.resize(ground_truth, (64,64))
        
        if ground_truth is None:
            print(image_path)
            print(gt_path)
            print("no gt")
            
            image_path = self.backup
            name = image_path.rsplit('/',1)[1]
            gt_split = name.replace('-','/').rsplit('_',1)
            gt_split = gt_split[0] + '/' + gt_split[1]
            if self.data_type == 'train':
                gt_path = "../DATA/GT/train/data_256/%s/%s" % (name[0],gt_split)
            else:
                gt_path = "../DATA/GT/val/val_256/%s" % name
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            ground_truth = cv2.imread(gt_path,cv2.IMREAD_COLOR)
            ground_truth = cv2.resize(ground_truth, (64,64))
        else:
            self.backup = image_path
            
        #noise_idx = np.random.randint(0,3,size =1)
        #image = addnoise(self.noise_type[noise_idx], image)

        if self.augmentation:
            # Not implemented yet
            # augmentation(image)
            pass
        
        if self.normalize:
            image = image / 255.0
            ground_truth = ground_truth / 255.0
            # image = (image/127.5) - 1
            # ground_truth = (ground_truth/127.5) - 1

        image = image.transpose(2,0,1)
        ground_truth = ground_truth.transpose(2,0,1)

        return torch.FloatTensor(image), torch.FloatTensor(ground_truth), name

