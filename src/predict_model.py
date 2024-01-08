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
new_image_paths = ["data/raw/images_dog_small/n02110958-pug/n02110958_353.jpg", "data/raw/images_dog_small/n02110185-Siberian_husky/n02110185_699.jpg"]  # Replace with actual file paths
new_images = [transform(Image.open(path)).to(device) for path in new_image_paths]

# Run inference on new images
model.eval()
with torch.no_grad():
    for i, new_image in enumerate(new_images):
        output = model(new_image.unsqueeze(0))  # Add batch dimension
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_probabilities, top5_classes = torch.topk(probabilities, 5)

        # Get the predicted class labels from the label encoder
        predicted_labels = label_encoder.inverse_transform(top5_classes.cpu().numpy())

        # Plot the image and predicted class with top 5 probabilities
        image = transforms.ToPILImage()(new_image.cpu()).convert("RGB")
        plt.imshow(image)
        plt.title(f"Prediction for image {i + 1}: Top 5 Classes and Probabilities")
        plt.axis('off')

        plt.show()
        print(predicted_labels)
        print(top5_probabilities)
