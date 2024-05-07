# imports
import torch
from ViT import ViT
from Preprocessing import *
from tqdm import tqdm
import albumentations as A

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
        self.LoadDatasets()
        self.LoadModel()

    def LoadModel(self):
        path = 'Transformer/Models/'
        filename = f'{self.dataset.__name__}{self.img_size}.pth'

        #self.model = ViT(out_dim=).to(self.device)
        self.model.load_state_dict(torch.load(path + filename))

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
        for subset in self.subsets:
            self.datasets[subset] = MedMNISTDataset(self.dataset, transform=transformer, dataset_type=subset, img_size=self.img_size, augment_data=self.augment_data)
            print(f'{subset} dataset loaded.')

    def ClassifyDatasets(self):
        self.model.eval()
        self.predictions = {}
        for subset in self.subsets:
            self.predictions[subset] = []
            for image, label in tqdm(self.datasets[subset], desc=f'Classifying {subset} dataset'):
                image = image.to(self.device)
                prediction = self.model(image)
                self.predictions[subset].append(prediction)
        
        return self.predictions

    def ClassifyImage(self, image):
        self.model.eval()
        image = image.to(self.device)
        prediction = self.model(image)
        return prediction
    

class EvaluationMetrics:
    def __init__(self):
        pass
