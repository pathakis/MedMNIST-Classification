from ViT import *
from Classfier import *
from Optimiser import *
from Preprocessing import *
from medmnist import PneumoniaMNIST, RetinaMNIST

def OptimiseViT(dataset, augment_data, epochs):
    '''
    Optimise the Vision Transformer model.
    '''
    optimiser = ViT_Optimiser(dataset, augment_data=augment_data, img_size=224)
    optimiser.RunOptimiser(epochs)

def ClassifyImage(dataset, image):
    '''
    Classify an image using the Vision Transformer model
    '''
    classifier = ViT_Classifier(dataset, ['test'], augment_data=True, img_size=224)
    classifier.LoadDatasets()
    predictions = classifier.ClassifyDatasets()
    print(predictions)

ClassifyImage(RetinaMNIST, None)

