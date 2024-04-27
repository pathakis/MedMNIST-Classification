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
    def __init__(self, dataset, transform=None, dataset_type='train', img_size=224, augment_data=False, balance_classes=False):
        self.dataset = dataset(split=dataset_type, download=True, size=img_size, as_rgb=True)
        self.transform = transform
        self.augment_data = augment_data
        if augment_data == False:
            self.Transform()
        if balance_classes:
            self.BalanceClasses()

    def Transform(self):
        tempDataset = []
        for idx, (image, label) in enumerate(self.dataset):
            if self.transform is not None:
                image = self.transform(image)
            tempDataset.append((image, label))
        self.dataset = tempDataset

    def BalanceClasses(self):
        '''
        Balance the classes in the dataset.
        '''
        if not self.augment_data:
            raise Exception("Cannot balance classes without augmenting the data.")
        
        # Get the number of samples in each class
        num_samples = {}
        for _, label in self.dataset:
            label = label[0]
            if label not in num_samples:
                num_samples[label] = 0
            num_samples[label] += 1
        print('Before balancing: ', num_samples)

        # Find the class with the most samples
        max_samples = max(num_samples.values())
        print('Max samples: ', max_samples)
        balanced_dataset = []

        # Find indicies of each class
        class_indices = {}
        for label in num_samples:
            class_indices[label] = [idx for idx, (_, l) in enumerate(self.dataset) if l == label]
        #print('Class indices: ', class_indices)
        # Select indicies to resample
        for label in num_samples.keys():
            print(f'Label 2: {label}', type(label))
            if num_samples[label] < max_samples:
                class_indices[label] = np.random.choice(class_indices[label], max_samples-len(class_indices))

        # Resample the dataset
        for idx, (image, label) in enumerate(self.dataset):
            balanced_dataset.append((image, label))
            if num_samples[label] < max_samples:
                if idx in class_indices[label]:
                    image = np.asarray(image)
                    image = self.transform(image=image)["image"]
                    if type(image) != torch.Tensor:
                        image = ToTensor()(image)
                    balanced_dataset.append((image, label))
        self.dataset = balanced_dataset
        
        num_samples = {}
        for _, label in self.dataset:
            if label not in num_samples:
                num_samples[label] = 0
            num_samples[label] += 1
        print('After balancing: ', num_samples)



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.augment_data and type(image) != torch.Tensor:
            image = np.asarray(image)
            image = self.transform(image=image)["image"]
            if type(image) != torch.Tensor:
                image = ToTensor()(image)
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

