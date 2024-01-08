import os
from PIL import Image
import torch
from torchvision import transforms
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def load_images(root_folder):
    image_list = []

    # Define a transformation to resize the images to 224x224 and convert them to a PyTorch tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Iterate through subfolders in the root folder
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            # Check if the file is an image file
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Construct the full path to the image
                image_path = os.path.join(subdir, file)

                try:
                    # Open the image using PIL
                    img = Image.open(image_path)

                    # Convert to RGB mode if not already in RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Apply the transformation
                    img_tensor = transform(img)

                    # Check if the image has 3 channels
                    if img_tensor.size(0) == 3:
                        # Add the image tensor to the list
                        image_list.append(img_tensor)
                    else:
                        print(f"Skipping image {image_path} due to incorrect number of channels.")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    if not image_list:
        raise RuntimeError("No valid images found in the specified folder.")

    # Stack the image tensors along the first dimension to create a single tensor
    images_tensor = torch.stack(image_list)

    return images_tensor

# Load images tensor as before
root_folder = "../../data/raw/images_dog"
images_tensor = load_images(root_folder)

# Print the size of the resulting tensor
print(images_tensor.size())


def extract_name_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    names = []

    for obj in root.findall(".//object"):
        name_element = obj.find("name")
        
        if name_element is not None and name_element.text is not None:
            names.append(name_element.text)

    return name_element.text

def process_folder(folder_path):
    res = []

    for folder in os.listdir(folder_path):
        if folder == ".DS_Store":
            continue
        
        xml_files = [file for file in os.listdir(os.path.join(folder_path, folder))]

        for file_name in xml_files:
            file_path = os.path.join(folder_path, folder, file_name)
            names = extract_name_from_xml(file_path)

            if names:
                #print(f"File: {file_name}, Object Names: {', '.join(names)}")
                res.append(names)
    return res

# Replace 'your_folder_path' with the path to your folder containing XML files without the ".xml" extension
folder_path = "../../data/raw/Annotation"

labels = process_folder(folder_path)

# Use LabelEncoder to convert string labels to numerical indices
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

print(labels_tensor.size())



# Normalize the data with mean 0 and standard deviation 1
mean_value = images_tensor.mean()
std_value = images_tensor.std()
train_images_tensor = (images_tensor - mean_value) / std_value

print("normal")

"""
n = images_tensor.shape[0]

# Reshape tensor2 to [n, 1] to make it compatible for concatenation
tensor2 = labels_tensor.view(n, 1)

# Concatenate the two tensors along the last dimension (axis=3)
merged_tensor = torch.cat((images_tensor, labels_tensor), dim=1)

indices = shuffle(np.arange(n))
"""

save_path = "../../data/processed"

# Save normalized tensors
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.save(train_images_tensor, os.path.join(save_path, 'train_images_tensor.pt'))

print("save1")

# Save labels tensor
torch.save(labels_tensor, os.path.join(save_path, 'train_target_tensor.pt'))

print("save2")

# Save label encoder
torch.save(label_encoder, os.path.join(save_path, 'label_encoder.pt'))