import albumentations as A
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import Dataset, DataLoader


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
    
class Augment:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image=image)["image"]
        return image
    
class MedMNISTDataset(Dataset):
    def __init__(self, dataset, transform=None, dataset_type='train', img_size=224, augment_data=False):
        self.dataset = dataset(split=dataset_type, download=True, size=img_size, as_rgb=True)
        self.transform = transform
        self.augment_data = augment_data
        self.Transform()

    def Transform(self):
        tempDataset = []
        for idx, (image, label) in enumerate(self.dataset):
            if self.transform is not None:
                if self.augment_data:
                    image = np.asarray(image)
                    image = self.transform(image=image)["image"]
                else:
                    image = self.transform(image)
            if type(image) != torch.Tensor:
                image = ToTensor()(image)
            tempDataset.append((image, label))
        self.dataset = tempDataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        '''
        if self.transform is not None and type(image) != torch.Tensor:
            if self.augment_data:
                #print('Augmenting data: ',type(image))
                image = np.asarray(image)
                image = self.transform(image=image)["image"]
            else:
                #print('Standard transform', type(image))
                image = self.transform(image)
        #print(type(image))
        if type(image) != torch.Tensor:
            image = ToTensor()(image)
            #print(' -> ', type(image))
        print(type(image))
        '''
        return image, label
    
class CatsVsDogsDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "Cat":
            label = 1.0
        else:
            label = 0.0
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

