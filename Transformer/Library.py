import os
import pickle
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ViT import *
from Tester import *
from torch.utils.data import DataLoader, random_split
from ViT import *
import torch
import tensorflow as tf
from einops import rearrange
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from einops import repeat
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ViT import *
from Tester import *
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