import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class CNN_MLP(nn.Module):
    def __init__(self):
        super(CNN_MLP, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 7 * 7, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x


# Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# Define dataloaders
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)


# Function to train the model

def train(model, criterion, optimizer, train_loader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")


# Function to test the model

def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {
          100 * correct / total}%')


# Function to visualize sample images

def visualize_samples(test_loader, model, num_samples):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            images = inputs.numpy()
            for j in range(num_samples):
                plt.imshow(np.squeeze(images[j]), cmap='gray')
                plt.title(f'Predicted: {predicted[j]}, Actual: {labels[j]}')
                plt.show()
            break


# Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Define the model
model = CNN_MLP()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
train(model, criterion, optimizer, train_loader, num_epochs)

# Test the model
test(model, test_loader)

# Visualize sample images
visualize_samples(test_loader, model, 5)
