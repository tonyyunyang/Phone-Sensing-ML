import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Additional imports
from torch.utils.mobile_optimizer import optimize_for_mobile

class SpectrogramDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
# Define data transformers
transform = transforms.Compose([
    transforms.Resize((20, 100)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# Create dataset
dataset = SpectrogramDataset(root="Training_data/New (06.02 added, update this name each time)", transform=transform)

# Create training, validation, and test splits
num_train = len(dataset)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))

np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]

train_size = len(train_idx)
validation_split = int(np.floor(0.1 * train_size))

np.random.shuffle(train_idx)
train_idx, valid_idx = train_idx[validation_split:], train_idx[:validation_split]

# Create data samplers and loaders
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

# Define batch size
batch_size_define = 32
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_define, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_define, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_define, sampler=test_sampler)

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
        self.rnn = nn.GRU(input_size=32*5*25, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, 7)  # Assuming 7 classes for location prediction

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1, x.shape[1]*x.shape[2]*x.shape[3])
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Instantiate the model and move it to GPU
model = CRNN()
model = model.to('cuda')

# Choose a loss function and optimizer
# Set the learning rate
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 25

# Training loop
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to('cuda')
        labels = labels.to('cuda')

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate running loss and accuracy
        train_running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    model.eval()
    valid_running_loss = 0.0
    valid_correct = 0
    valid_total = 0
    for images, labels in valid_loader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculate running loss and accuracy
        valid_running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        valid_total += labels.size(0)
        valid_correct += (predicted == labels).sum().item()

    valid_loss = valid_running_loss / len(valid_loader)
    valid_acc = 100 * valid_correct / valid_total

    # Append the losses and accuracies for this epoch to our lists
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.2f}%, '
          f'Validation Loss: {valid_loss:.4f}, '
          f'Validation Acc: {valid_acc:.2f}%')

print('Finished Training')

# Save the PyTorch model
torch.save(model.state_dict(), 'model.pth')

# Get the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the correct device
model.to(device)

# Ensure the input data is also on the correct device
example = torch.rand(1, 3, 20, 100)
example = example.to(device)

# Now you can trace your model
traced_script_module = torch.jit.trace(model, example)

optimized_torchscript_model = optimize_for_mobile(traced_script_module)
optimized_torchscript_model.save("model.ptl")

# Now, let's plot our losses and accuracies over epochs
epochs = range(1, num_epochs+1)

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'g', label='Training Loss')
plt.plot(epochs, valid_losses, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'g', label='Training Accuracy')
plt.plot(epochs, valid_accuracies, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
