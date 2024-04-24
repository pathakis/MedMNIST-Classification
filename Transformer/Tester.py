from ViT import *
import torch
import tensorflow as tf

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
    model = ViT()
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

GPUAccessTest()
