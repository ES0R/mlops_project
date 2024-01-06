import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

rawfolder = "../../data/raw/images"
save_path = "../../data/processed"

# Initialize empty lists to store train tensors
train_images_tensors = []

# Define the desired size for resizing
desired_size = (224, 224)  # Adjust this based on your requirements

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels if needed
    transforms.Resize(desired_size),
    transforms.ToTensor(),
])


# Loop through train files
for filename in os.listdir(rawfolder):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Assuming images are jpg, jpeg, or png
        try:
            # Load the image
            image_path = os.path.join(rawfolder, filename)
            image = Image.open(image_path)

            # Apply the transformation
            image_tensor = transform(image)

            # Add the tensor to the list
            train_images_tensors.append(image_tensor)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Concatenate tensors along the first dimension
train_images_tensor = torch.stack(train_images_tensors, dim=0)


# Normalize the data with mean 0 and standard deviation 1
mean_value = train_images_tensor.mean()
std_value = train_images_tensor.std()
train_images_tensor = (train_images_tensor - mean_value) / std_value

# Save normalized tensors
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.save(train_images_tensor, os.path.join(save_path, 'train_images_tensor.pt'))



