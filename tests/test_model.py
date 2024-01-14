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

from src.models.model import MyAwesomeModel, MyImprovedCNNModel
import numpy as np

def test_input_MyAwesomeModel():
    input_features = 3*224*224
    model = MyAwesomeModel()
    assert input_features == model.fc1.in_features, "number of input features does not match [3*224*224]"

def test_layers_MyAwesomeModel():
    model = MyAwesomeModel()
    assert model.fc1.out_features == model.fc2.in_features, "This model is not made correctly. Output and Input of subsequent layers do not match!"
    assert model.fc2.out_features == model.fc3.in_features, "This model is not made correctly. Output and Input of subsequent layers do not match!"
    assert model.fc3.out_features == model.fc4.in_features, "This model is not made correctly. Output and Input of subsequent layers do not match!"

def test_layers_MyImprovedCNNModel():
    model = MyImprovedCNNModel()
    assert model.fc1.out_features == model.fc2.in_features, "This model is not made correctly. Output and Input of subsequent layers do not match!"
    assert model.fc2.out_features == model.fc3.in_features, "This model is not made correctly. Output and Input of subsequent layers do not match!"






