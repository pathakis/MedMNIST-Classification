from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split
from ViT import *
from Tester import *

#GPUAccessTest()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f' > Using device: {device}\n')

training = PneumoniaMNIST(split='train', download=True, size=224, as_rgb=True, transform=Compose([Resize((224, 224)), ToTensor()]))
train_loader = DataLoader(training, batch_size=32, shuffle=True)
test = PneumoniaMNIST(split='test', download=True, size=224, as_rgb=True, transform=Compose([Resize((224, 224)), ToTensor()]))
test_loader = DataLoader(test, batch_size=32, shuffle=True)

model = ViT().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 10

for epoch in range(epochs):
    epoch_losses = []
    model.train()
    for step, (input, labels) in enumerate(train_loader):
        input, labels = input.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(input)
        labels_one_hot = F.one_hot(labels, num_classes=8).float()
        loss = criterion(output, labels_one_hot)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f"Epoch {epoch+1} - Training loss: {np.mean(epoch_losses)}")
    for step, (input, labels) in enumerate(test_loader):
        input, labels = input.to(device), labels.to(device)
        output = model(input)
        labels_one_hot = F.one_hot(labels, num_classes=8).float()
        loss = criterion(output, labels_one_hot)
        epoch_losses.append(loss.item())
    print(f"Epoch {epoch+1} - Testing loss: {np.mean(epoch_losses)}")
    
