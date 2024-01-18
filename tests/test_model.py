import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.model import MyImprovedCNNModel
import numpy as np

def test_layer1():
    model = MyImprovedCNNModel()
    assert model.fc1.out_features == model.fc2.in_features, "This model is not made correctly. Output and Input of subsequent layers do not match!"

def test_layer2():
    model = MyImprovedCNNModel()
    assert model.fc2.out_features == model.fc3.in_features, "This model is not made correctly. Output and Input of subsequent layers do not match!"

def test_layer3():
    model = MyImprovedCNNModel()
    assert model.batchnorm1.num_features == model.conv1.out_channels, "This model is not made correctly. Output and Input of subsequent layers do not match!"
    assert model.batchnorm2.num_features == model.conv2.out_channels, "This model is not made correctly. Output and Input of subsequent layers do not match!"
    assert model.batchnorm3.num_features == model.conv3.out_channels, "This model is not made correctly. Output and Input of subsequent layers do not match!"


