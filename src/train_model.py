import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from models.model import MyAwesomeModel
from models.model import MyCNNModel  # Assuming you save the CNN model in cnn_model.py
import matplotlib.pyplot as plt
import argparse
import hydra

split_index = 1000
model_now = "CNN"
epochs = 10

# Load your data using the mnist function from data.py
train_images_tensor = torch.load("../data/processed/train_images_tensor.pt")
train_target_tensor = torch.load("../data/processed/train_target_tensor.pt")


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


