# imports
import torch, pickle
from ViT import ViT
from Preprocessing import *
from tqdm import tqdm
import albumentations as A
from torch.utils.data import DataLoader

class ViT_Classifier:
    def __init__(self, dataset, subsets, augment_data=False, img_size=224):
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
        self.dataset = dataset
        self.subsets = subsets
        self.augment_data = augment_data


        # Model
        if len(subsets) > 0:
            self.LoadDatasets()
            #self.num_classes = len(self.datasets[self.subsets[0]].num_classes)
        else:
            pass
            '''self.subsets = ['test']
            self.LoadDatasets()'''
        self.LoadModel()

    def LoadModel(self):
        path = 'Transformer/Models/'
        filename = f'{self.dataset.__name__}{self.img_size}'

        with open(path + 'ModelInfo.pkl', 'rb') as file:
            self.modelInfo = pickle.load(file)

        self.model = ViT(out_dim=self.modelInfo[filename]['Classes']).to(self.device)

        self.model.load_state_dict(torch.load(path + filename+'.pth'))

    

    def LoadDatasets(self):
        if self.augment_data:
            print("Augmenting data...")
            transformer = A.Compose([
                    A.Rotate(limit=30, p=0.5),              # Rotate the image by up to 30 degrees with a probability of 0.5
                    A.RandomScale(scale_limit=0.2, p=0.5),  # Randomly scale the image by up to 20% with a probability of 0.5
                    A.RandomBrightnessContrast(p=0.5),      # Randomly adjust brightness and contrast with a probability of 0.5
                    A.GaussianBlur(p=0.5),                  # Apply Gaussian blur with a probability of 0.5
                    A.HorizontalFlip(p=0.5),                # Flip the image horizontally with a probability of 0.5
                    A.VerticalFlip(p=0.5),                  # Flip the image vertically with a probability of 0.5
                    A.GridDistortion(p=0.5),                # Apply grid distortion with a probability of 0.5
                    A.Resize(height=self.img_size, width=self.img_size),   # Resize the image to the desired size
                    A.Normalize(),                          # Normalize the image                             # Convert the image to a PyTorch tensor
                    ])
        else:
            transformer = Compose([
                Resize((self.img_size, self.img_size)), 
                ToTensor()]
                )
        self.datasets = {}
        self.loaders = {}
        for subset in self.subsets:
            self.datasets[subset] = MedMNISTDataset(self.dataset, transform=transformer, dataset_type=subset, img_size=self.img_size, augment_data=self.augment_data)
            self.loaders[subset] = DataLoader(self.datasets[subset], batch_size=1, shuffle=False)
            print(f'{subset} dataset loaded.')

    def ClassifyDatasets(self):
        self.model.eval()
        self.predictions = {}
        self.true_labels = {}
        for subset in self.subsets:
            self.predictions[subset] = []
            self.true_labels[subset] = []
            for step, (image, label) in tqdm(enumerate(self.loaders[subset]), desc=f'Classifying {subset} dataset'):
                image, labels = image.to(self.device), label.to(self.device)
                prediction = self.model(image)
                prediction = torch.argmax(prediction, dim=1)
                self.predictions[subset].append(prediction.detach().cpu())
                self.true_labels[subset].append(labels.detach().cpu())
        
        return self.predictions, self.true_labels

    def ClassifyImage(self, image):
        self.model.eval()
        local_transform = Compose([Resize((self.img_size, self.img_size)), ToTensor()])
        image = local_transform(image).to(self.device)
        loader = DataLoader([(image, 0)], batch_size=1, shuffle=False)
        for step, (image, label) in enumerate(loader):
            prediction = self.model(image)
            prediction = torch.argmax(prediction, dim=1)
        return prediction.detach().cpu()
    

class EvaluationMetrics:
    def __init__(self, predictions, true_labels):
        self.predictions = predictions
        self.true_labels = true_labels

    def Accuracy(self):
        accuracies = {}
        for subset in self.predictions:
            correct = 0
            total = 0
            for i, prediction in enumerate(self.predictions[subset]):
                correct += (prediction == self.true_labels[subset][i]).sum().item()
                total += len(prediction)
            accuracies[subset] = correct / total
        return accuracies
    
    def ConfusionMatrix(self):
        confusion_matrices = {}
        for subset in self.predictions:
            confusion_matrix = torch.zeros(5, 5)
            for i, prediction in enumerate(self.predictions[subset]):
                for t, true_label in enumerate(self.true_labels[subset][i]):
                    confusion_matrix[true_label, prediction[t]] += 1
            confusion_matrices[subset] = confusion_matrix
        return confusion_matrices
    
    def Precision(self):
        precision = {}
        for subset in self.predictions:
            confusion_matrix = self.ConfusionMatrix()[subset]
            precision[subset] = confusion_matrix.diag() / confusion_matrix.sum(0)
        return precision
    
    def Recall(self):
        recall = {}
        for subset in self.predictions:
            confusion_matrix = self.ConfusionMatrix()[subset]
            recall[subset] = confusion_matrix.diag() / confusion_matrix.sum(1)
        return recall
    
    def F1Score(self):
        f1 = {}
        for subset in self.predictions:
            precision = self.Precision()[subset]
            recall = self.Recall()[subset]
            f1[subset] = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def PrintMetrics(self):
        accuracies = self.Accuracy()
        precisions = self.Precision()
        recalls = self.Recall()
        f1 = self.F1Score()
        for subset in self.predictions:
            print(f'{subset} dataset:')
            print(f' > Accuracy: {accuracies[subset]}')
            print(f' > Precision: {precisions[subset]}')
            print(f' > Recall: {recalls[subset]}')
            print(f' > F1 Score: {f1[subset]}')
            print((confusion_matrix := self.ConfusionMatrix()[subset]).numpy())
            print('\n')