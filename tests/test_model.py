import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from src.models.model import MyCNNModel 
import numpy as np

def test_my_cnn_model():
    # Initialize your CNN model
    model = MyCNNModel()

    # Set the model to evaluation mode
    model.eval()

    # Define a sample image path
    sample_image_path = "data/raw/images/image_1.jpg"

    # Load and preprocess the image
    image = Image.open(sample_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Assuming your model expects a 224x224 image
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_image)

    # Perform assertions
    # For example, you can check if the output has the expected shape
    assert output.shape == torch.Size([1, 5]), f"Expected output shape: [1, 5], Actual shape: {output.shape}"

    # You can also check if the output probabilities sum to 1 (if your model outputs probabilities)
    assert torch.allclose(output.sum(dim=1), torch.tensor(1.0)), "Output probabilities do not sum to 1"

    





