from Library import *
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms import Resize, ToTensor
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
import torcheval.metrics

class ViT_Optimiser:
    def __init__(self, dataset, model=None, optimizer=None, trainingcriterion=None, testcriterion=None):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f' > Using device: {self.device}\n')
        if self.device == "cpu":
            print("WARNING: MPS not available, using CPU instead.")
            if input("Continue? (y/n): ") == "n":
                exit()

        # Load training and validation data
        self.LoadDatasets(dataset)
        
        self.model = ViT().to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.trainingCriterion = nn.CrossEntropyLoss()

        #self.testCriterion = nn.Accuracy()

    def LoadDatasets(self, dataset):
        self.training = dataset(split='train', download=True, size=224, as_rgb=True, transform=Compose([Resize((224, 224)), ToTensor()]))
        self.train_loader = DataLoader(self.training, batch_size=32, shuffle=True)

        self.validation = dataset(split='val', download=True, size=224, as_rgb=True, transform=Compose([Resize((224, 224)), ToTensor()]))
        self.validation_loader = DataLoader(self.validation, batch_size=32, shuffle=True)

        self.test = dataset(split='test', download=True, size=224, as_rgb=True, transform=Compose([Resize((224, 224)), ToTensor()]))
        self.test_loader = DataLoader(self.test, batch_size=32, shuffle=True)
    
    def RunOptimiser(self, epochs, title):
        print(f"Running optimiser for {epochs} epochs on {title} dataset...")

        for epoch in range(epochs):
            epoch_losses = { "training": [], "validation": []}
            self.model.train()

            # Optimise model on training data
            for step, (input, labels) in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch+1}", total=len(self.train_loader)):
                input, labels = input.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.trainingCriterion(output, labels.squeeze())
                loss.backward()
                self.optimizer.step()
                epoch_losses["training"].append(loss.item())

            # Run model over validation data
            for step, (input, labels) in enumerate(self.validation_loader):
                input, labels = input.to(self.device), labels.to(self.device)
                output = self.model(input)
                loss = self.trainingCriterion(output, labels.squeeze())
                epoch_losses["validation"].append(loss.item())
        
            print(f"\nEpoch {epoch+1}\n   - Training loss: {np.mean(epoch_losses['training'])}\n   - Validation loss: {np.mean(epoch_losses['validation'])}\n\n")

        for step, (input, labels) in enumerate(self.test_loader):
            input, labels = input.to(self.device), labels.to(self.device)
            output = self.model(input)
            loss = self.trainingCriterion(output, labels.squeeze())
        print(f"Epoch {epoch+1} - Testing loss: {loss.item()}")