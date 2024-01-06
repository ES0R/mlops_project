import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

rawfolder = "data/raw/images"
labels_file = "data/raw/labels.csv"
save_path = "data/processed"

# Load labels from CSV
labels_df = pd.read_csv(labels_file)
labels_column = 'overall_sentiment'
labels = labels_df[labels_column].tolist()

# Use LabelEncoder to convert string labels to numerical indices
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

# Initialize empty lists to store train tensors and corresponding labels
train_images_tensors = []
corresponding_labels = []

# Define the desired size for resizing
desired_size = (224, 224)  # Adjust this based on your requirements

transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.ToTensor(),
])


print("Start loading images")

# Loop through train files
for i,filename in enumerate(os.listdir(rawfolder)):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Assuming images are jpg, jpeg, or png
        try:
            # Load the image
            image_path = os.path.join(rawfolder, filename)
            image = Image.open(image_path)

            # Convert to RGB if the image is grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply the transformation
            image_tensor = transform(image)

            # Add the tensor to the list
            train_images_tensors.append(image_tensor)

            # Get corresponding label and add it to the list
            corresponding_label = labels_df.loc[labels_df['image_name'] == filename][labels_column].values[0]
            corresponding_labels.append(label_encoder.transform([corresponding_label])[0])
            if (i % 500) == 0:
                print(i)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            print(i)



# Convert the list of labels to a tensor
corresponding_labels_tensor = torch.tensor(corresponding_labels, dtype=torch.long)

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

# Save labels tensor
torch.save(corresponding_labels_tensor, os.path.join(save_path, 'train_target_tensor.pt'))

# Save label encoder
torch.save(label_encoder, os.path.join(save_path, 'label_encoder.pt'))

print(f"Images loaded: {corresponding_labels_tensor.size()}")

