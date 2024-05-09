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


class ViTOptimiser:
    def __init__(self, dataset, img_size=224, augment_data=False, increaseSize=0, balance_classes=False, vit_patch_size=4, batch_size=32):
        # Set the device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == "cpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cpu":
                print("WARNING: MPS not available, using CPU instead.")
                if input("Continue? (y/n): ") != "y":
                    exit()
        print(f"Using device: {self.device}")

        # General parameters
        self.img_size = int(img_size)
        self.patch_size = vit_patch_size
        self.augment_data = augment_data
        self.sample_size = increaseSize
        self.balance_classes = balance_classes

        # Load datasets
        self.filename = f'{dataset.__name__}{self.img_size}'
        self.batch_size = batch_size
        self.dataset = dataset
        self.LoadDatasets()

        # Load or create model
        self.LoadModelInformation()
        self.model = ViT(img_size=img_size,         # Image size
                         patch_size=vit_patch_size, # Patch size, resolution at which the ViT model processes the image. Smaller gives higher precision at a larger computational cost.
                         out_dim=self.num_classes   # Number of output classes
                         ).to(self.device)          # Assign to GPU
        self.LoadViT()

        # Training parameters
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)                  # Optimiser
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)      # Alternative optimiser
        self.criterion = nn.CrossEntropyLoss()                                          # Loss function

    def LoadDatasets(self):
        '''
        Load the datasets and assign transformer.
        '''
        self.datasets = {}
        self.loaders = {}

        augmentingTransformer = A.Compose([
                    A.Rotate(limit=30, p=0.),              # Rotate the image by up to 30 degrees with a probability of 0.5
                    A.RandomScale(scale_limit=0.2, p=0.),  # Randomly scale the image by up to 20% with a probability of 0.5
                    A.RandomBrightnessContrast(p=0.),      # Randomly adjust brightness and contrast with a probability of 0.5
                    #A.GaussianBlur(p=0.5),                  # Apply Gaussian blur with a probability of 0.5
                    A.HorizontalFlip(p=0.),                # Flip the image horizontally with a probability of 0.5
                    A.VerticalFlip(p=0.),                  # Flip the image vertically with a probability of 0.5
                    #A.GridDistortion(p=0.5),                # Apply grid distortion with a probability of 0.5
                    A.Resize(height=self.img_size, width=self.img_size),   # Resize the image to the desired size
                    A.Normalize(),                          # Normalize the image                             # Convert the image to a PyTorch tensor
                    ])
        standardTransformer = A.Compose([
                    A.Resize(height=self.img_size, width=self.img_size),   # Resize the image to the desired size
                    ])
        #standardTransformer = Compose([
        #            Resize((self.img_size, self.img_size)), # Set image size
        #            ToTensor()                              # Convert the image to a PyTorch tensor
        #            ])
        
        for subset in ['train', 'val', 'test']:
            if self.augment_data:
                transformer = augmentingTransformer
            else:
                transformer = standardTransformer
            
            if subset == 'test':
                transformer = standardTransformer
                balance = False
                sample_size = 0
            else:
                balance = self.balance_classes
                sample_size = self.sample_size
                if subset == 'val':
                    sample_size *= 0.1

            self.datasets[subset] = MedMNISTDataset(self.dataset, 
                                                    transform=transformer, 
                                                    dataset_type=subset, 
                                                    img_size=self.img_size, 
                                                    augment_data=self.augment_data, 
                                                    balance_classes=balance, 
                                                    nSamples=sample_size
                                                    )
            self.loaders[subset] = DataLoader(self.datasets[subset], 
                                              batch_size=self.batch_size, 
                                              shuffle=True)
        self.num_classes = self.datasets['train'].num_classes
            
    def LoadModelInformation(self):
        '''
        Load the model information. It will contain the performance metrics for the dataset as well as some general model information.
        '''
        path = 'Transformer/Models/'
        if not os.path.exists(path):
            os.makedirs(path)
        modelInfoPath = path + f'ModelInfo.pkl'
        if os.path.exists(modelInfoPath):
            with open(modelInfoPath, 'rb') as f:
                self.modelInfo = pickle.load(f)

            if self.filename not in self.modelInfo:
                self.modelInfo[self.filename] = {'Training': {'Accuracy': 0, 'F1': 0, 'Loss': 1000}, 
                                                 'Validation': {'Accuracy': 0, 'F1': 0, 'Loss': 1000}, 
                                                 'Test': {'Accuracy': 0, 'F1': 0, 'Loss': 1000}, 
                                                 'Model': 'ViT', 
                                                 'Loss function': 'CrossEntropyLoss', 
                                                 'Classes': self.num_classes}
        else:
            self.modelInfo = {}
            self.modelInfo[self.filename] = {'Training': {'Accuracy': 0, 'F1': 0, 'Loss': 1000}, 
                                             'Validation': {'Accuracy': 0, 'F1': 0, 'Loss': 1000}, 
                                             'Test': {'Accuracy': 0, 'F1': 0, 'Loss': 1000}, 
                                             'Model': 'ViT', 
                                             'Loss function': 'CrossEntropyLoss', 
                                             'Classes': self.num_classes}
    
    def LoadViT(self):
        '''
        Load the Vision Transformer model.
        '''
        path = 'Transformer/Models/'
        try:
            self.model.load_state_dict(torch.load(path + f'{self.filename}' + '.pth'))
            print(f'\nModel loaded from {path + f"{self.filename}" + ".pkl"}.')
            for key in self.modelInfo[self.filename]:
                print(f'{key}: {self.modelInfo[self.filename][key]}')
        except:
            print('No model found, training new model...')

    def SaveModelInformation(self):
        '''
        Save the model information.
        '''
        path = 'Transformer/Models/'
        with open(path + f'ModelInfo.pkl', 'wb') as f:
            pickle.dump(self.modelInfo, f)

    def SaveViT(self):
        '''
        Save the Vision Transformer model.
        '''
        directory = 'Transformer/Models'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, self.filename + '.pth')
        
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to '{filepath}'.\n")

    def EvaluationMetrics(self, output, labels, losses):
        '''
        Calculate the evaluation metrics.
        '''
        output_np = np.array(output)
        labels_np = np.array(labels)
        output_np = np.argmax(output_np, axis=1)
        accuracy = accuracy_score(labels_np, output_np)
        f1 = f1_score(labels_np, output_np, average='macro')
        return {'Accuracy': accuracy, 'F1': f1, 'Loss': np.mean(losses)}

    def RunOptimiser(self, epochs, verboseInterval=5):
        '''
        Run the optimiser.
        '''
        print(f'\nTraining model {self.filename} for {epochs} epochs...\n')
        self.model.train()

        for epoch in range(epochs):
            epoch_losses = {'training': [], 'validation': []}
            epoch_output = []
            epoch_labels = []

            # Training
            for (images, labels) in tqdm(self.loaders['train'], desc=f'Epoch {epoch}/{epochs}', total=len(self.loaders['train'])):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels.squeeze())
                loss.backward()
                self.optimizer.step()
                epoch_losses['training'].append(loss.item())
                epoch_output += output.tolist()
                epoch_labels += labels.squeeze().tolist()
            
            trainingPerformance = self.EvaluationMetrics(epoch_output, epoch_labels, epoch_losses['training'])
            if epoch % verboseInterval == 0:

                # Validation
                for (images, labels) in self.loaders['val']:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self.model(images)
                    loss = self.criterion(output, labels.squeeze())
                    epoch_losses['validation'].append(loss.item())
                    epoch_output += output.tolist()
                    epoch_labels += labels.squeeze().tolist()
                validationPerformance = self.EvaluationMetrics(epoch_output, epoch_labels, epoch_losses['validation'])

                print(f'Epoch {epoch}/{epochs}:')
                print(f'   Training set:')
                print(f'      - Loss: {np.mean(epoch_losses["training"]):.4f} | Accuracy: {trainingPerformance["Accuracy"]} | F1: {trainingPerformance["F1"]:.2f}\n')
                print(f'   Validation set:')
                print(f'      - Loss: {np.mean(epoch_losses["validation"]):.4f} | Accuracy: {validationPerformance["Accuracy"]} | F1: {validationPerformance["F1"]:.2f}\n')

                if validationPerformance['Accuracy'] > self.modelInfo[self.filename]['Validation']['Accuracy']:
                    print('Accuracy increased on validation set, saving model...')
                    self.modelInfo[self.filename]['Training'] = trainingPerformance
                    self.modelInfo[self.filename]['Validation'] = validationPerformance
                    testPerformance = self.RunTest(verbose=False)
                    self.modelInfo[self.filename]['Test'] = testPerformance
                    self.SaveModelInformation()
                    self.SaveViT()
        
        print(f'\nTraining complete.\n')
        testPerformance = self.RunTest(verbose=True)
        print(f'\nModel performance:')
        print(f'   Test set:')
        print(f'      - Loss: {testPerformance["Loss"]:.4f} | Accuracy: {testPerformance["Accuracy"]:.2f} | F1: {testPerformance["F1"]:.2f}\n')

    def RunTest(self, verbose=False):
        '''
        Run the test.
        '''
        self.model.eval()
        test_losses = []
        test_output = []
        test_labels = []

        for (images, labels) in self.loaders['test']:
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            loss = self.criterion(output, labels.squeeze())
            test_losses.append(loss.item())
            test_output += output.tolist()
            test_labels += labels.squeeze().tolist()

        testPerformance = self.EvaluationMetrics(test_output, test_labels, test_losses)
        print(f'Test set:')
        print(f'   - Loss: {np.mean(test_losses):.2f} | Accuracy: {testPerformance["Accuracy"]:.2f} | F1: {testPerformance["F1"]:.2f}\n')
        print(f'Model output:')

        if verbose:
            print(f'\nModel performance:')
            for i, pred in enumerate(output):
                pred = pred.detach().cpu().numpy()
                print(f'   - Prediction: {np.argmax(pred)} | Truth: {test_labels[i]}')
                if i == 10:
                    break

        return testPerformance


    
