from medmnist import PneumoniaMNIST, RetinaMNIST, ChestMNIST
#from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image
import albumentations
from albumentations import Compose as ACompose
from sklearn.utils import resample

class Augmenter:
    def __init__(self) -> None:
        pass

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(image=img)['image']
        return img
    
class Augment:
    '''
    Class for augmenting images using the Albumentations library. It will only augment a specific image and not affect class balance or dataset size.
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(image=img)['image']
        return img

# Example usage:
help(ACompose)
        