from ViT import *
from Classfier import *
from Optimiser import *
from Preprocessing import *
from medmnist import PneumoniaMNIST, RetinaMNIST

def OptimiseViT(dataset, augment_data, epochs, img_size=224):
    '''
    Optimise the Vision Transformer model.
    '''
    optimiser = ViT_Optimiser(dataset, augment_data=augment_data,img_size=img_size)
    optimiser.RunOptimiser(epochs)

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
augment_data = False
img_size = 28
run_type = 'classify_dataset' # 'optimise', 'classify_image', 'classify_dataset'

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

if run_type == 'optimise':
    OptimiseViT(dataset, augment_data,epochs, img_size)
elif run_type == 'classify_image':
    prediction = ClassifyImage(classifier, image)
    print(f'Prediction: {prediction}')
elif run_type == 'classify_dataset':
    ClassifyDataset(classifier)


