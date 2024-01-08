import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from models.model import MyAwesomeModel
from models.model import MyCNNModel, UNet, SimpleCNNModel   # Assuming you save the CNN model in cnn_model.py
import matplotlib.pyplot as plt
import argparse
import hydra
from tqdm import tqdm

split_index = 150
model_now = "CNN"
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your data using the mnist function from data.py
train_images_tensor = torch.load("data/processed/train_images_tensor.pt")
train_target_tensor = torch.load("data/processed/train_target_tensor.pt")

# Ensure that the data tensors have the same length
assert len(train_images_tensor) == len(train_target_tensor), "Mismatch in data length"

print(train_images_tensor.size())
print(train_target_tensor.size())

# Choose the model based on the command-line argument
if model_now == 'FCNN':
    model = MyAwesomeModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif model_now == 'CNN':
    model = MyCNNModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif model_now == 'UNet':
    model = UNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif model_now == 'Simple':
    model = SimpleCNNModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
else:
    raise ValueError('Invalid model choice.')


# Hyperparameters
batch_size = 64

# Create a DataLoader for the entire dataset
full_dataset = TensorDataset(train_images_tensor, train_target_tensor)
full_loader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=True)

# Split the shuffled dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create a DataLoader for training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create a DataLoader for validation data
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



print("Training begins")

# Lists to store values for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    # Use tqdm for the training loop
    for batch_images, batch_targets in tqdm(train_loader, desc=f'Training - Epoch [{epoch + 1}/{epochs}]'):
        batch_images = batch_images.to(device)
        batch_targets = batch_targets.to(device)
        outputs = model(batch_images)

        loss = criterion(outputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += batch_targets.size(0)
        correct_train += (predicted_train == batch_targets).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy_train = correct_train / total_train
    print(f'Training - Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy_train:.4f}')

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    # Use tqdm for the validation loop
    for batch_images_val, batch_targets_val in tqdm(val_loader, desc=f'Validation - Epoch [{epoch + 1}/{epochs}]'):
        batch_images_val = batch_images_val.to(device)
        batch_targets_val = batch_targets_val.to(device)

        outputs_val = model(batch_images_val)
        loss_val = criterion(outputs_val, batch_targets_val)
        val_loss += loss_val.item()

        _, predicted_val = torch.max(outputs_val.data, 1)
        total_val += batch_targets_val.size(0)
        correct_val += (predicted_val == batch_targets_val).sum().item()

    average_val_loss = val_loss / len(val_loader)
    accuracy_val = correct_val / total_val
    print(f'Validation - Epoch [{epoch + 1}/{epochs}], Loss: {average_val_loss:.4f}, Accuracy: {accuracy_val:.4f}')

    # Save values for plotting
    train_losses.append(average_loss)
    val_losses.append(average_val_loss)
    train_accuracies.append(accuracy_train)
    val_accuracies.append(accuracy_val)

# Plotting
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 4))

# Plotting training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()


# Optionally, save your trained model
torch.save(model.state_dict(), 'models/trained_model.pth')
