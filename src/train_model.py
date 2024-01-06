import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from models.model import MyAwesomeModel
from models.model import MyCNNModel   # Assuming you save the CNN model in cnn_model.py
import matplotlib.pyplot as plt
import argparse
import hydra

split_index = 1000
model_now = "CNN"
epochs = 10

# Load your data using the mnist function from data.py
train_images_tensor = torch.load("../data/processed/train_images_tensor.pt")
train_target_tensor = torch.load("../data/processed/train_target_tensor.pt")

# Ensure that the data tensors have the same length
assert len(train_images_tensor) == len(train_target_tensor), "Mismatch in data length"

print(train_images_tensor.size())
print(train_target_tensor.size())

train_images_tensor, val_images_tensor = (
    train_images_tensor[split_index:],
    train_images_tensor[:split_index],
)
train_target_tensor, val_target_tensor = (
    train_target_tensor[split_index:],
    train_target_tensor[:split_index],
)

# Choose the model based on the command-line argument
if model_now == 'FCNN':
    model = MyAwesomeModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif model_now == 'CNN':
    model = MyCNNModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
else:
    raise ValueError('Invalid model choice. Use --model FCNN or --model CNN')


# Assuming MyAwesomeModel or MyCNNModel has a forward method
# and you have defined train_images_tensor, train_target_tensor, criterion, and optimizer as mentioned in your code

# Hyperparameters
batch_size = 64

# Create a DataLoader for training data
train_dataset = TensorDataset(train_images_tensor, train_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create a DataLoader for validation data
val_dataset = TensorDataset(val_images_tensor, val_target_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_images, batch_targets in train_loader:
        # Forward pass
        outputs = model(batch_images)

        # Compute the loss
        loss = criterion(outputs, batch_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average training loss for the epoch
    average_loss = total_loss / len(train_loader)
    print(f'Training - Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for batch_images_val, batch_targets_val in val_loader:
            outputs_val = model(batch_images_val)
            loss_val = criterion(outputs_val, batch_targets_val)
            val_loss += loss_val.item()

            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += batch_targets_val.size(0)
            correct_val += (predicted_val == batch_targets_val).sum().item()

    # Print the average validation loss and accuracy for the epoch
    average_val_loss = val_loss / len(val_loader)
    accuracy_val = correct_val / total_val
    print(f'Validation - Epoch [{epoch + 1}/{epochs}], Loss: {average_val_loss:.4f}, Accuracy: {accuracy_val:.4f}')

# Optionally, save your trained model
# torch.save(model.state_dict(), 'trained_model.pth')
