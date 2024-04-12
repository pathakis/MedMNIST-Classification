from medmnist import PneumoniaMNIST, RetinaMNIST
import pickle, os

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