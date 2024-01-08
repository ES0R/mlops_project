import torch
from torch.utils.data import DataLoader
from models.model import MyCNNModel  # Update the import statement if necessary
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
model = MyCNNModel()
model.load_state_dict(torch.load('models/trained_model.pth'))
model.eval()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transform function for new images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the label encoder
label_encoder = torch.load('data/processed/label_encoder.pt')

# Load and preprocess new images
new_image_paths = ["data/raw/images_dog_small/n02105412-kelpie/n02105412_123.jpg"]  # Replace with actual file paths
new_images = [transform(Image.open(path)).to(device) for path in new_image_paths]

# Run inference on new images
model.eval()
with torch.no_grad():
    for i, new_image in enumerate(new_images):
        output = model(new_image.unsqueeze(0))  # Add batch dimension
        _, predicted_class = torch.max(output, 1)
        
        # Get the predicted class label from the label encoder
        predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
        
        # Plot the image and predicted class
        image = transforms.ToPILImage()(new_image.cpu()).convert("RGB")
        plt.imshow(image)
        plt.title(f"Prediction for image {i+1}: Class {predicted_label}")
        plt.show()
