from Preprocessing import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from ViT import *
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
import albumentations as A
import numpy as np
import os
import pickle
import torch
import torch.optim as optim

class ViT_Optimiser:
    def __init__(self, dataset, img_size=224, augment_data=False):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f' > Using device: {self.device}\n')
        if self.device == "cpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cpu":
                print("WARNING: MPS not available, using CPU instead.")
                if input("Continue? (y/n): ") != "y":
                    exit()

        # Parameters
        self.img_size = int(img_size)
        self.augment_data = augment_data

        # Load training and validation data
        self.LoadDatasets(dataset)
        self.dataset = dataset
        self.LoadPerformance()
        print(self.modelPerformance)
        if str(dataset.__name__) not in self.modelPerformance:
            self.modelPerformance[str(dataset.__name__)] = {'Training': {'Accuracy': 0, 'F1': 0}, 'Validation': {'Accuracy': 0, 'F1': 0}, 'Model': 'ViT', 'Loss function': 'CrossEntropyLoss'}
            print(self.modelPerformance)
            self.SavePerformance()

        # Define model
        self.model = ViT(out_dim=self.num_classes).to(self.device)
        try:
            if str(dataset.__name__) in self.modelPerformance:
                self.LoadModel(dataset.__name__)
        except:
            print("No model found, training new model...")
        
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
        self.training = MedMNISTDataset(dataset, transform=trainingTransformer, dataset_type='train', img_size=self.img_size, augment_data=self.augment_data, balance_classes=True)
        self.train_loader = DataLoader(self.training, batch_size=32, shuffle=True)
        self.num_classes = self.training.num_classes

        self.validation = MedMNISTDataset(dataset, transform=standardTransformer, dataset_type='val', img_size=self.img_size)
        self.validation_loader = DataLoader(self.validation, batch_size=32, shuffle=True)

        self.test = MedMNISTDataset(dataset, transform=standardTransformer, dataset_type='test', img_size=self.img_size)
        self.test_loader = DataLoader(self.test, batch_size=32, shuffle=True)
    
    def RunOptimiser(self, epochs):
        print(f"Running optimiser for {epochs} epochs on {str(self.dataset.__name__)} dataset...")

        for epoch in range(epochs):
            epoch_losses = { "training": [], "validation": []}
            self.model.train()
            collected_training_output = []
            collected_validation_output = []
            collected_test_output = []
            collected_training_labels = []
            collected_validation_labels = []
            collected_test_labels = []

            # Optimise model on training data
            for step, (input, labels) in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch+1}", total=len(self.train_loader)):
                input, labels = input.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                #print(input.shape)
                output = self.model(input)
                loss = self.trainingCriterion(output, labels.squeeze())
                collected_training_output += output.tolist()
                collected_training_labels += labels.squeeze().tolist()
                loss.backward()
                self.optimizer.step()
                epoch_losses["training"].append(loss.item())
            
            trainingPerformance = self.EvaluatePerformance(collected_training_output, collected_training_labels)

            # Run model over validation data
            for step, (input, labels) in enumerate(self.validation_loader):
                input, labels = input.to(self.device), labels.to(self.device)
                output = self.model(input)
                collected_validation_output += output.tolist()
                collected_validation_labels += labels.squeeze().tolist()
                loss = self.trainingCriterion(output, labels.squeeze())
                epoch_losses["validation"].append(loss.item())

            validationPerformance = self.EvaluatePerformance(collected_validation_output, collected_validation_labels)

            print(f'\nEpoch {epoch+1}/{epochs}\n')
            print(f'   Training set:\n')
            print(f'      - Loss: {np.mean(epoch_losses["training"]):.2f} | Accuracy: {trainingPerformance["Accuracy"]:.2f} | F1: {trainingPerformance["F1"]:.2f}\n')
            print(f'   Validation set:\n')
            print(f'      - Loss: {np.mean(epoch_losses["validation"]):.2f} | Accuracy: {validationPerformance["Accuracy"]:.2f} | F1: {validationPerformance["F1"]:.2f}\n')

            # If model has better performance on validation set than previous runs, save model
            if validationPerformance["Accuracy"] > self.modelPerformance[str(self.dataset.__name__)]['Validation']['Accuracy']:
                self.modelPerformance[str(self.dataset.__name__)]['Training'] = trainingPerformance
                self.modelPerformance[str(self.dataset.__name__)]['Validation'] = validationPerformance
                self.SavePerformance()
                self.SaveModel(self.dataset.__name__)
        
            #print(f"\nEpoch {epoch+1}\n   - Training loss: {np.mean(epoch_losses['training'])}\n   - Validation loss: {np.mean(epoch_losses['validation'])}\n\n")

        for step, (input, labels) in enumerate(self.test_loader):
            input, labels = input.to(self.device), labels.to(self.device)
            output = self.model(input)
            collected_test_output += output.tolist()
            collected_test_labels += labels.squeeze().tolist()
            loss = self.trainingCriterion(output, labels.squeeze())

        testPerformance = self.EvaluatePerformance(collected_test_output, collected_test_labels)
        print(f'   Test set:')
        print(f'      - Loss: {loss.item():.2f} | Accuracy: {testPerformance["Accuracy"]:.2f} | F1: {testPerformance["F1"]:.2f}')

    def EvaluatePerformance(self, output, labels):
        # Make predictions available on CPU
        output_np = np.array(output)
        labels_np = np.array(labels)
        #output_np = output.detach().cpu().numpy()
        #labels_np = labels.detach().cpu().numpy()
        output_np = np.argmax(output_np, axis=1)

        # Calculate performance metrics
        accuracy = accuracy_score(labels_np, output_np)                 # Calculate accuracy
        f1 = f1_score(labels_np, output_np, average='macro')            # Calculate F1 score
        return {'Accuracy': accuracy, 'F1': f1}

    def SaveModel(self, filename):
        print("Saving model...")
        directory = 'Transformer/Models'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename + '.pth')
        
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to '{filepath}'.")

    def LoadModel(self, filename):
        print("Loading model...")
        path = 'Transformer/Models/' + filename + '.pth'
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from '{path}'.")

    def SavePerformance(self):
        path = 'Transformer/Models/Performance.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self.modelPerformance, file)

    def LoadPerformance(self):
        with open('Transformer/Models/Performance.pkl', 'rb') as file:
            self.modelPerformance = pickle.load(file)
        