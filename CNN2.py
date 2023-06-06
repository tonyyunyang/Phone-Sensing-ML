import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Define paths and parameters
base_dir = 'Training_data/New (06.06 added, update this name each time)'
learning_rate = 0.0005
num_epochs = 50
batch_size = 64

# Check if CUDA is available and set PyTorch to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data transformations
data_transforms = transforms.Compose([
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)) # normalize to range [-1, 1]
])

# Load the datasets
full_dataset = datasets.ImageFolder(root=base_dir, transform=data_transforms)

# Split the dataset into training, validation and test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

num_classes = len(full_dataset.classes)

# Define the model
# model = nn.Sequential(
#     nn.Conv2d(3, 32, 3, 1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(32, 64, 3, 1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Flatten(),
#     nn.Linear(64 * 5 * 25, 128),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(128, num_classes)
# ).to(device) # Move the model to GPU

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 125, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out
model = CNN().to(device) # Move the model to GPU


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Lists to track loss and accuracy
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Training loop
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_correct = 0
    val_correct = 0
    
    # Training
    for inputs, labels in train_loader:
        # Move inputs and labels to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
    
    train_losses.append(train_loss / len(train_dataset))
    train_acc = 100.0 * train_correct / len(train_dataset)
    train_accs.append(train_acc)
    
    # Validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            
    val_losses.append(val_loss / len(val_dataset))
    val_acc = 100.0 * val_correct / len(val_dataset)
    val_accs.append(val_acc)
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    print(f'Train loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2f}')
    print(f'Val loss: {val_losses[-1]:.4f}, Acc: {val_acc:.2f}')
    print('-' * 10)

# Testing
test_correct = 0
# Initialize lists for predicted labels and true labels
all_predicted = []
all_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        
        # Collect predicted and true labels
        all_predicted.extend(predicted.cpu().numpy())
        all_true.extend(labels.cpu().numpy())

test_acc = 100.0 * test_correct / len(test_dataset)
print('Test Accuracy: {:.2f}'.format(test_acc))

# Plot the training and validation loss
fig, axs = plt.subplots(2)
fig.suptitle('Training Metrics')

axs[0].plot(train_losses, label='Training loss')
axs[0].plot(val_losses, label='Validation loss')
axs[0].set_title('Loss')
axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axs[0].legend()

axs[1].plot(train_accs, label='Training accuracy')
axs[1].plot(val_accs, label='Validation accuracy')
axs[1].set_title('Accuracy')
axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axs[1].legend()

plt.show()

# Convert the lists of predicted and true labels to numpy arrays
all_predicted = np.array(all_predicted)
all_true = np.array(all_true)

# Compute the confusion matrix
confusion_mat = confusion_matrix(all_true, all_predicted)
total_per_class = confusion_mat.sum(axis=1)  # Sum of each row (total true labels per class)
percentage_confusion_mat = confusion_mat / total_per_class[:, np.newaxis]  # Convert counts to percentages

# Plot the confusion matrix with percentages
class_labels = ["C" + str(i + 1) for i in range(num_classes)]  # Create a list of class labels
plt.figure(figsize=(10, 8))
sns.heatmap(percentage_confusion_mat, annot=True, fmt=".2%", cmap="Blues", cbar=False)
plt.title("Confusion Matrix (Percentages)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(ticks=np.arange(num_classes), labels=class_labels)
plt.yticks(ticks=np.arange(num_classes), labels=class_labels)
plt.show()

print('Finished Training')
