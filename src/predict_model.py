import hydra
from omegaconf import DictConfig
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.models.model import CustomCNN, MyImprovedCNNModel,ViTModel  # Adjust path according to your project structure
import pytorch_lightning as pl
from google.cloud import storage
import yaml
HYDRA_FULL_ERROR=1


def list_bucket_items(local_path):
    client = storage.Client()
    bucket_name = "mlops-doggy"
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs()

    namelist = []
    timelist = []
    for blob in blobs:
        namelist.append(blob.name)
        timelist.append(blob.time_created)

    # Combine namelist and timelist into tuples
    combined_list = list(zip(namelist, timelist))

    # Sort the combined list based on the values in timelist
    sorted_list = sorted(combined_list, key=lambda x: x[1])

    # Unpack the sorted list back into namelist and timelist
    namelist, timelist = zip(*sorted_list)

    blob = bucket.blob(namelist[-1])
    blob.download_to_filename(local_path)
        




class ImageClassifier(pl.LightningModule):
    def __init__(self, cfg, num_classes):
        super(ImageClassifier, self).__init__()
        self.cfg = cfg

        if cfg.default_model == 'cnn':
            self.model = CustomCNN(cfg, num_classes)
        elif cfg.default_model == 'vit':
            self.model = ViTModel(cfg, num_classes)
        else:
            raise ValueError("Unsupported model type specified in configuration")

        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

# Function to read yaml configuration (this might be optional if you're using Hydra)
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to preprocess the image
def preprocess_image(image: Image.Image, device, transform):
    """
    Preprocess the image to be suitable for the model.
    """
    return transform(image).to(device)

# Function to predict image
def predict_image(image: Image.Image, model, device, label_encoder, transform):
    """
    Run model prediction on the preprocessed image and return top 5 labels and probabilities.
    """
    model.eval()
    processed_image = preprocess_image(image, device, transform)
    with torch.no_grad():
        output = model(processed_image.unsqueeze(0))  # Add batch dimension
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_probabilities, top5_classes = torch.topk(probabilities, 3)
        
        print("Known classes in label encoder:", label_encoder.classes_, top5_classes)#-torch.tensor([1,1,1,1,1]))
        # Get the predicted class labels from the label encoder
        predicted_labels = label_encoder.inverse_transform(top5_classes.cpu().numpy())
    
    return predicted_labels, top5_probabilities.cpu().numpy()

# Function to select random images
def select_random_images(base_path, num_images=2):
    # List all subdirectories in the base path
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    selected_images = []

    for _ in range(num_images):
        # Randomly select a subdirectory
        subdir = random.choice(subdirs)
        subdir_path = os.path.join(base_path, subdir)

        # List all files in the selected subdirectory
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        if files:
            # Randomly select a file
            file = random.choice(files)
            selected_images.append(os.path.join(subdir_path, file))

    return selected_images

# Main prediction function using Hydra
@hydra.main(version_base=None, config_path="config", config_name="config")
def predict(cfg: DictConfig):
    # Load the configuration


    config = read_yaml('src/config/data/data_config.yaml')
    class_mapping = read_yaml('data/class_mapping.yaml')
    selected_classes = {class_mapping[int(cls)] for cls in config.get('classes', [])}
        
    # Load the saved model
    model = ImageClassifier(cfg.model, len(selected_classes))
     
    model_local_path = 'models/model-trained-temp.pth'

    list_bucket_items(model_local_path)
    
    model.load_state_dict(torch.load(model_local_path))
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
    print(label_encoder.classes_)
    
    # Load and preprocess new images
    base_path = 'data/raw/images/Images'
    random_images = select_random_images(base_path, num_images=2)
    new_images = [(Image.open(path)) for path in random_images]

    # Run inference on new images
    for i, image in enumerate(new_images):
        predicted_labels, top5_probabilities = predict_image(image, model, device, label_encoder, transform)
        plt.imshow(image)
        plt.title(f"Prediction for image {i + 1}: Top 5 Classes and Probabilities")
        plt.axis('off')
        plt.show()
        print(predicted_labels)
        print(top5_probabilities)

if __name__ == "__main__":
    predict()
