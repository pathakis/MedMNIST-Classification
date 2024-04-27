import albumentations as A
from torchvision.transforms import Resize, ToTensor

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
    
class Dataset:
    def __init__(self, dataset, transform, dataset_type='train',img_size=224):
        # Parameters
        self.img_size = int(img_size)
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.transform = transform
        self.LoadDatasets()

    def LoadDatasets(self):
        dataset = self.dataset(split=self.dataset_type, download=True, size=self.img_size, as_rgb=True, transform=self.transform)
        self.images = dataset['images']
        self.labels = dataset['labels']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

