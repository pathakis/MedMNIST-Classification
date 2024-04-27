# We will store all the functions that are used in the main file here. This will help us keep the main file clean and easy to read.
#from Optimiser import ViT_Optimiser
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