import torch
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTForImageClassification
from medmnist import PneumoniaMNIST, RetinaMNIST
import pickle, os
import numpy as np
import pandas as pd
import sklearn, torch
import tensorflow as tf

def Preprocess(batch):
    inputs = featureExtractor(batch['img'], return_tensors='pt')
    inputs['labels'] = batch['labels']
    return inputs

def GetExtractor():
    path = 'google/vit-base-patch16-224-in21k'
    featureExtractor = ViTImageProcessor.from_pretrained(path)
    print('Successfully loaded feature extractor.')
    return featureExtractor

def GPUAccessTest():
    # Check for TensorFlow GPU access
    print(f"\nTensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    # See TensorFlow version
    print(f" > TensorFlow version: {tf.__version__}")
    print(f' > Pytorch version', torch.__version__)
    print(f' > Is MPS built? {torch.backends.mps.is_built()}')
    print(f' > Is MPS available? {torch.backends.mps.is_available()}')

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f' > Using device: {device}\n')

    x = torch.rand(size=(3,4)).to(device)
    print(f" > Tensor: {x}")

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

def SeperateData(dataset):
    images = []
    labels = []
    for (img, label) in dataset:
        images.append(img)
        labels.append(label)
    return images, labels


def Collate(batchImages, batchLabels):
    batchImages = torch.stack([x[0] for x in batchImages])
    batchLabels = torch.tensor(x for x in batchLabels)
    return {'pixel_values': batchImages, 'labels': batchLabels}
