from Library import *

retinaTraining = RetinaMNIST(split='train', download=True, size=224)
retinaValidation = RetinaMNIST(split='val', download=True, size=224)
retinaTest = RetinaMNIST(split='test', download=True, size=224)

imagesTraining, labelsTraining = SeperateData(retinaTraining)
imagesValidation, labelsValidation = SeperateData(retinaValidation)
imagesTest, labelsTest = SeperateData(retinaTest)
print('Successfully loaded RetinaMNIST datasets.')

processor = GetExtractor()
featuresTraining = processor(imagesTraining, return_tensors='pt')
featuresValidation = processor(imagesValidation, return_tensors='pt')
featuresTest = processor(imagesTest, return_tensors='pt')
print('Successufly extracted features from datasets.')






