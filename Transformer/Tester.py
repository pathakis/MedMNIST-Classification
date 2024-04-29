# Written by Oscar Rosman
# Date: 2024-04-24
# Program to test the ViT model and its components.

from Library import *
from ViT import *
from Optimiser import ViT_Optimiser
from medmnist import PneumoniaMNIST, RetinaMNIST, ChestMNIST
import time


def RunViT_Test():
    # Test normalisation class / layer
    print('\nTesting normalisation PreNorm class...')
    norm = PreNorm(64, Attention(64, 8, 0.1))
    print(norm(torch.ones(10, 64, 64)).shape)
    print('Normalisation test passed.\n\n\n')

    # Test feed forward class
    print('Testing feed forward class...')
    ff = FeedForward(64, 128)
    print(ff(torch.ones(10, 64, 64)).shape)
    print('Feed forward test passed.\n\n\n')

    # Test residual attention class
    print('Testing residual attention class...')
    residual_att = ResidualAdd(Attention(64, 8, 0.))
    print(residual_att(torch.ones(10, 64, 64)).shape)
    print('Residual attention test passed.\n\n\n')

    # Test patch embedding
    print('Testing patch embedding...')
    to_tensor = [Resize((224, 224)), ToTensor()]
    dataset = PneumoniaMNIST(split='train', download=True, size=224, as_rgb=True,transform=Compose(to_tensor))
    sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
    print("Initial shape: ", sample_datapoint.shape)
    print(sample_datapoint)
    embedding = PatchEmbedding()(sample_datapoint)
    print("Patches shape: ", embedding.shape)
    print('Patch embedding test passed.\n\n\n')

    # Test ViT model
    print('Testing ViT model...')
    model = ViT(out_dim=5)
    print(model)
    print(model(torch.rand(1, 3, 224, 224)))
    print('ViT model test passed.')

def GPUAccessTest():
    '''
    Check access to GPU devices for TensorFlow and PyTorch, might only work on macOS.
    '''
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

def RunOptimisationTest(dataset, augment,balance_classes, epochs=2):
    '''
    Test the ViT optimiser class.
    '''
    optimiser = ViT_Optimiser(dataset, augment_data=augment, img_size=224)
    optimiser.RunOptimiser(epochs)

def SaveModelTest():
    '''
    Test saving and loading a model.
    '''
    trainer = ViT_Optimiser(RetinaMNIST, augment_data=True)
    trainer.RunOptimiser(2)
    model = trainer.model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.save(model.state_dict(), 'Transformer/Models/RetinaModel.pth')
    print(f'Model saved on device {device}. At location: Transformers/Models/RetinaModel.pth.')

def LoadModelTest():
    '''
    Test loading a model.
    '''
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ViT().to(device)
    model.load_state_dict(torch.load('Transformer/Models/RetinaModel.pth'))
    model
    print('Model loaded.')
    trainer = ViT_Optimiser(RetinaMNIST, 2)
    trainer.model = model
    trainer.RunOptimiser(2)
    print('Model loaded and tested.')

def IntegratedSaveLoadTest(mode='save'):
    '''
    Test saving and loading a model using the integrated functions.
    '''
    if mode == 'save':
        trainer = ViT_Optimiser(RetinaMNIST, augment_data=False, img_size=224)
        trainer.RunOptimiser(2)
        trainer.SaveModel(trainer.dataset.__name__)
        print('Model saved succesfully.')
    elif mode == 'Load':
        trainer = ViT_Optimiser(RetinaMNIST, augment_data=True)
        trainer.LoadModel(trainer.dataset.__name__)
        trainer.RunOptimiser(2)
        print('Model loaded succesfully.')


if __name__ == '__main__':
    st = time.time()
    #RunViT_Test()
    #GPUAccessTest()
    RunOptimisationTest(RetinaMNIST, True, True, 2)
    #SaveModelTest()
    #LoadModelTest()
    #IntegratedSaveLoadTest('save')
    #IntegratedSaveLoadTest('Load')

    print('\n'*3,'#'*50)
    print(f'\n\n\nTests completed in {(time.time() - st)//60:.1f} minutes and {(time.time() - st)%60} seconds.\n\n\n')

RunOptimisationTest(RetinaMNIST, True, True, 150)