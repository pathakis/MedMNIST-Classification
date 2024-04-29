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
    
class MedMNISTDataset(Dataset):
    def __init__(self, dataset, transform=None, dataset_type='train', img_size=224, nSamples=0,augment_data=False, balance_classes=False):
        self.dataset = dataset(split=dataset_type, download=True, size=img_size, as_rgb=True)
        self.transform = transform
        self.augment_data = augment_data
        self.increaseSize = nSamples
        if augment_data == False:
            self.Transform()

        if balance_classes:
            self.BalanceClasses(nSamples, verbose=True)

    def Transform(self):
        tempDataset = []
        for idx, (image, label) in enumerate(self.dataset):
            if self.transform is not None:
                image = self.transform(image)
            tempDataset.append((image, label))
        self.dataset = tempDataset

    def BalanceClasses(self, increaseSize=0, verbose=False):
        '''
        Balance the classes in the dataset by resampling the minority classes. For the best efftect
        augmentation should be enabled. Otherwise the same image will be duplicated within the dataset.
        '''
        print('Balancing classes...')
        
        # Get the number of samples in each class
        num_samples = {}
        for _, label in self.dataset:
            label = label[0]
            #print(f'Label: {label}', type(label))
            if label not in num_samples:
                num_samples[label] = 0
            num_samples[label] += 1
        self.num_classes = len(num_samples)
        if verbose:
            print(f'Before balancing: {num_samples} | Num classes: {self.num_classes}')

        # Find the class with the most samples
        if increaseSize > 0:
            if int(increaseSize / len(num_samples)) < max(num_samples.values()):
                max_samples = max(num_samples.values())
            else:
                max_samples = int(increaseSize / len(num_samples))
                print(len(num_samples), increaseSize, increaseSize / len(num_samples), int(increaseSize / len(num_samples)))
                print(f'Increasing size to {max_samples} samples per class.')
        else:
            max_samples = max(num_samples.values())
        
        # Create a balanced dataset
        balanced_dataset = []
        for image, label in self.dataset:
            balanced_dataset.append((image, label))

        # Resample minority classes
        for label, count in num_samples.items():
            if count < max_samples:
                # Number of samples to add
                num_to_add = max_samples - count

                # Get indices of samples in the minority class
                class_indices = [idx for idx, (_, l) in enumerate(self.dataset) if l == label]

                # Select indices to resample
                selected_indices = np.random.choice(class_indices, num_to_add, replace=True)

                for idx in selected_indices:
                    image, label = self.dataset[idx]
                    balanced_dataset.append((image, label))
        self.dataset = balanced_dataset
        
        # Control the balance
        if verbose:
            num_samples = {}
            for _, label in self.dataset:
                label = label[0]
                if label not in num_samples:
                    num_samples[label] = 0
                num_samples[label] += 1
            print('After balancing: ', num_samples)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        '''
        Get an item from the dataset, if augmention is enabled, augment the data.
        '''
        image, label = self.dataset[idx]
        if self.augment_data and type(image) != torch.Tensor:
            image = np.asarray(image)
            image = self.transform(image=image)["image"]
            if type(image) != torch.Tensor:
                image = ToTensor()(image)
        return image, label
    

