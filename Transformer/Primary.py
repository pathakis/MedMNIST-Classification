from ViT import *
from Classfier import *
from Optimiser import *
from NewOptimiser import *
from Preprocessing import *
from medmnist import PneumoniaMNIST, RetinaMNIST
import time

def OptimiseViT(dataset, augment, balance, epochs, img_size=224, increaseSize=0, batch_size=32):
    '''
    Optimise the Vision Transformer model.
    '''
    # Old optimiser
    #optimiser = ViT_Optimiser(dataset, augment_data=augment_data,img_size=img_size, increaseSize=dataset_size)
    #optimiser.RunOptimiser(epochs)

    # New optimiser
    optimiser = ViTOptimiser(dataset, 
                             augment_data=augment,
                             increaseSize=increaseSize,
                             balance_classes=balance,
                             img_size=img_size, 
                             batch_size=batch_size)
    
    optimiser.RunOptimiser(epochs, verboseInterval=2)

def ClassifyImage(classifier, image):
    '''
    Classify an image using the Vision Transformer model
    '''
    prediction = classifier.ClassifyImage(image)
    return prediction.item()

def ClassifyDataset(classifier):
    '''
    Classify a dataset using the Vision Transformer model.
    '''
    predictions, truth = classifier.ClassifyDatasets()
    metrics = EvaluationMetrics(predictions, truth)
    metrics.PrintMetrics()

##################################################################################################
# Change parameters here
dataset = PneumoniaMNIST
epochs = 200
subsets = ['train', 'test'] # 'train', 'val', 'test'
augment_data = True
balance_classes = True
dataset_size_increase = 10000
img_size = 128
batch_size = 128
run_type = 'optimise' # 'optimise', 'classify_image', 'classify_dataset'

if run_type == 'classify_image':
    # Set the image here
    imgExtract = MedMNISTDataset(dataset, dataset_type='test', img_size=img_size, augment_data=augment_data)
    image, _ = imgExtract.__getitem__(0)

##################################################################################################
# Loads the classifier, do not touch
if run_type != 'optimise':
    classifier = ViT_Classifier(dataset, subsets, augment_data=augment_data, img_size=img_size)
    classifier.LoadDatasets()

##################################################################################################
# Modify the code below to test the classifier
st = time.time()
if run_type == 'optimise':
    OptimiseViT(dataset, augment_data, balance_classes, epochs, img_size, dataset_size_increase, batch_size)
elif run_type == 'classify_image':
    prediction = ClassifyImage(classifier, image)
    print(f'Prediction: {prediction}')
elif run_type == 'classify_dataset':
    ClassifyDataset(classifier)

print(f'Execution time: {(time.time() - st)//60:.0f} minutes and {(time.time() - st)%60:.0f} seconds.')


