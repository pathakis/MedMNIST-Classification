# Trash program to save old functions and classes that are no longer used but have a potential future use.
from ViT import *
from Optimiser import ViT_Optimiser
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
from medmnist import PneumoniaMNIST
from medmnist import PneumoniaMNIST, RetinaMNIST, ChestMNIST
from random import random
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torcheval.metrics

def SaveData(dataset, filename):
    directory = 'Data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename + '.pkl')
    
    with open(filepath, 'wb') as file:
        pickle.dump(dataset, file)
    
    print(f"Data saved to '{filepath}'.")

def LoadData(filename):
    path = 'Data/' + filename + '.pkl'
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    print(f"Data loaded from '{path}'.")
    return dataset

class Augment:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image=image)["image"]
        return image
    
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