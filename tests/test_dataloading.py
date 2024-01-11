import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data.make_dataset import load_images

def test_length():
    train_images_tensor = torch.load("data/processed/train_images_tensor.pt")
    train_target_tensor = torch.load("data/processed/train_target_tensor.pt")

    assert len(train_images_tensor) == len(train_target_tensor), "Mismatch in data length!"

def test_size():
    train_images_tensor = torch.load("data/processed/train_images_tensor.pt")
    test_tensor = torch.randn(3, 224, 224)
    assert train_images_tensor.shape[-3:] == test_tensor.shape, "Images are not the shape of [3, 224, 224]!"

def test_dimension_tensors():
    train_images_tensor = torch.load("data/processed/train_images_tensor.pt")
    train_target_tensor = torch.load("data/processed/train_target_tensor.pt")

    assert train_images_tensor.dim() == 4, "Image tensor is not 4-dimensional!"
    assert train_target_tensor.dim() == 1, "Label tensor is not 1-dimensional!"
