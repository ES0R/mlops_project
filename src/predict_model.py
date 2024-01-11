import torch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.models.model import MyImprovedCNNModel  # Update the import statement if necessary
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
model = MyImprovedCNNModel()
model.load_state_dict(torch.load('models/trained_model_ADV_.pth'))
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

def preprocess_image(image: Image.Image):
    """
    Preprocess the image to be suitable for the model.
    """
    return transform(image).to(device)

def predict_image(image: Image.Image):
    """
    Run model prediction on the preprocessed image and return top 5 labels and probabilities.
    """
    model.eval()
    processed_image = preprocess_image(image)
    with torch.no_grad():
        output = model(processed_image.unsqueeze(0))  # Add batch dimension
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_probabilities, top5_classes = torch.topk(probabilities, 5)
        
        # Get the predicted class labels from the label encoder
        predicted_labels = label_encoder.inverse_transform(top5_classes.cpu().numpy())
    
    return predicted_labels, top5_probabilities.cpu().numpy()

if __name__ == "__main__":
    # Load and preprocess new images
    new_image_paths = ["data/raw/images_dog_small/n02110958-pug/n02110958_353.jpg", "data/raw/images_dog_small/n02110185-Siberian_husky/n02110185_699.jpg"]  # Replace with actual file paths
    new_images = [Image.open(path) for path in new_image_paths]

    # Run inference on new images
    for i, image in enumerate(new_images):
        predicted_labels, top5_probabilities = predict_image(image)

        # Plot the image and predicted class with top 5 probabilities
        plt.imshow(image)
        plt.title(f"Prediction for image {i + 1}: Top 5 Classes and Probabilities")
        plt.axis('off')

        plt.show()
        print(predicted_labels)
        print(top5_probabilities)
