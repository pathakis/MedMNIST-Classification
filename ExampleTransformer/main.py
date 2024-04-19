import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTForImageClassification
from LoadData import *
import numpy as np
import torch

penumoniaDataSet = LoadData('PneumoniaDataSet')
retinaDataSet = LoadData('RetinaDataSet')

path = 'google/vit-base-patch16-224-in21k'
featureExtractor = ViTImageProcessor.from_pretrained(path)
print('Successfully loaded feature extractor.')

# Plot or show the image
image = penumoniaDataSet.trainingSet['img'][0]
# Convert image to numpy array
image_array = np.array(image)

# Print the RGB matrix
#print(image_array.shape)

# Show the image
example = featureExtractor(penumoniaDataSet.trainingSet['img'][0], return_tensors='pt')
#print(example['pixel_values'].shape)

# Check access to GPU
GPUAccessTest()


