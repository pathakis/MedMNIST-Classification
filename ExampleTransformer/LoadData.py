from medmnist import PneumoniaMNIST, RetinaMNIST
import pickle, os
import numpy as np
import pandas as pd
import sklearn, torch
import tensorflow as tf

class DataSet:
    def __init__(self, datasetMNIST):
        self.trainingSet = datasetMNIST(split='train', download=True)
        self.validationSet = datasetMNIST(split='val', download=True)
        self.testSet = datasetMNIST(split='test', download=True)

    def OverwriteSubsets(self):
        self.trainingSet = SplitData(self.trainingSet)
        self.validationSet = SplitData(self.validationSet)
        self.testSet = SplitData(self.testSet)
        

def SplitData(subset):
    if type(subset) == dict:
        print("Already split")
        return subset
    
    images = []
    labels = []
    for (img, label) in subset:
        images.append(img.convert('RGB'))
        labels.append(label)
    output = dict()
    output['img'] = images
    output['labels'] = labels

    return output

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

def GPUAccessTest():
    # Check for TensorFlow GPU access
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    # See TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f'Pytorch version', torch.__version__)
    print(f'Is MPS built? {torch.backends.mps.is_built()}')
    print(f'Is MPS available? {torch.backends.mps.is_available()}')

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'Using device: {device}')



