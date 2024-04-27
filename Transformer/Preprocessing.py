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
        print('Before balancing: ', num_samples)

        # Find the class with the most samples
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

