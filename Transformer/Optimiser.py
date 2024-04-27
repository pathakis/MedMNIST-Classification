from Preprocessing import *
from ViT import *
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
import albumentations as A
import numpy as np
import os
import torch
import torch.optim as optim

class ViT_Optimiser:
    def __init__(self, dataset, img_size=224, augment_data=False, model=None, optimizer=None, trainingcriterion=None, testcriterion=None):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f' > Using device: {self.device}\n')
        if self.device == "cpu":
            print("WARNING: MPS not available, using CPU instead.")
            if input("Continue? (y/n): ") != "y":
                exit()

        # Parameters
        self.img_size = int(img_size)
        self.augment_data = augment_data
        print(self.img_size)


        # Load training and validation data
        self.LoadDatasets(dataset)
        self.dataset = dataset
        
        self.model = ViT().to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.trainingCriterion = nn.CrossEntropyLoss()

        #self.testCriterion = nn.Accuracy()
    
    def LoadDatasets(self, dataset):
        if self.augment_data:
            print("Augmenting data...")
            trainingTransformer = A.Compose([
                    A.Rotate(limit=30, p=0.5),              # Rotate the image by up to 30 degrees with a probability of 0.5
                    A.RandomScale(scale_limit=0.2, p=0.5),  # Randomly scale the image by up to 20% with a probability of 0.5
                    A.RandomBrightnessContrast(p=0.5),      # Randomly adjust brightness and contrast with a probability of 0.5
                    A.GaussianBlur(p=0.5),                  # Apply Gaussian blur with a probability of 0.5
                    #A.RandomNoise(p=0.5),                   # Add random noise with a probability of 0.5
                    A.HorizontalFlip(p=0.5),                # Flip the image horizontally with a probability of 0.5
                    A.VerticalFlip(p=0.5),                  # Flip the image vertically with a probability of 0.5
                    #A.RandomCrop(height=224, width=224),    # Randomly crop the image to size 224x224
                    A.GridDistortion(p=0.5),                # Apply grid distortion with a probability of 0.5
                    A.Resize(height=self.img_size, width=self.img_size),   # Resize the image to the desired size
                    A.Normalize(),                          # Normalize the image                             # Convert the image to a PyTorch tensor
                    ])
        else:
            trainingTransformer = Compose([
                Resize((self.img_size, self.img_size)), 
                ToTensor()]
                )
        standardTransformer = Compose([Resize((self.img_size, self.img_size)), ToTensor()])
        self.training = MedMNISTDataset(dataset, transform=trainingTransformer, dataset_type='train', img_size=self.img_size, augment_data=self.augment_data)
        self.train_loader = DataLoader(self.training, batch_size=32, shuffle=True)

        self.validation = MedMNISTDataset(dataset, transform=standardTransformer, dataset_type='val', img_size=self.img_size)
        self.validation_loader = DataLoader(self.validation, batch_size=32, shuffle=True)

        self.test = MedMNISTDataset(dataset, transform=standardTransformer, dataset_type='test', img_size=self.img_size)
        self.test_loader = DataLoader(self.test, batch_size=32, shuffle=True)
        '''
        self.training = dataset(split='train', download=True, size=self.img_size, as_rgb=True, transform=trainingTransformer)
        self.train_loader = DataLoader(self.training, batch_size=32, shuffle=True)
        print('Augmentation complete')
        self.validation = dataset(split='val', download=True, size=self.img_size, as_rgb=True, transform=Compose([Resize((self.img_size, self.img_size)), ToTensor()]))
        self.validation_loader = DataLoader(self.validation, batch_size=32, shuffle=True)

        self.test = dataset(split='test', download=True, size=self.img_size, as_rgb=True, transform=Compose([Resize((self.img_size, self.img_size)), ToTensor()]))
        self.test_loader = DataLoader(self.test, batch_size=32, shuffle=True)
        print('Data loaded')
        '''
    
    def RunOptimiser(self, epochs):
        print(f"Running optimiser for {epochs} epochs on {str(self.dataset.__name__)} dataset...")

        for epoch in range(epochs):
            epoch_losses = { "training": [], "validation": []}
            self.model.train()

            # Optimise model on training data
            for step, (input, labels) in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch+1}", total=len(self.train_loader)):
                input, labels = input.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                #print(input.shape)
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

    def SaveModel(self, filename):
        directory = 'Transformer/Models'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename + '.pth')
        
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to '{filepath}'.")

    def LoadModel(self, filename):
        path = 'Transformer/Models/' + filename + '.pth'
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from '{path}'.")