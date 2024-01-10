import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Path to the saved tensors and label encoder
save_path = "data/processed"
tensor_filename = "train_images_tensor.pt"
labels_filename = "train_target_tensor.pt"
label_encoder_filename = "label_encoder.pt"

# Load the saved tensors and label encoder
loaded_tensor = torch.load(os.path.join(save_path, tensor_filename))
labels_tensor = torch.load(os.path.join(save_path, labels_filename))
label_encoder = torch.load(os.path.join(save_path, label_encoder_filename))

# Get the number of examples in the dataset
num_examples = len(loaded_tensor)

# Randomly select 5 indices
random_indices = torch.randperm(num_examples)[:5]

# Use the random indices to extract 5 random example images and labels
example_images = loaded_tensor[random_indices]
example_labels = labels_tensor[random_indices]

# Define a function to display images with corresponding labels and class mapping
def show_images_with_labels(images, labels, label_encoder):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))

    for i, (image, label) in enumerate(zip(images, labels)):
        # Move channels to the last dimension for displaying
        img = image.permute(1, 2, 0).numpy()

        # Ensure values are in the range [0, 1]
        img = (img - img.min()) / (img.max() - img.min())

        axes[i].imshow(img)
        axes[i].axis('off')

        # Decode label using label_encoder
        decoded_label = label_encoder.classes_[label.item()]

        axes[i].set_title(f"Class: {decoded_label} ({label.item()})")

    plt.show()

# Display the 5 random example images with corresponding labels and class mapping
show_images_with_labels(example_images, example_labels, label_encoder)

# Print the mapping of class numbers to strings
class_mapping = dict(enumerate(label_encoder.classes_))
print("Class Mapping:", class_mapping)
print(labels_tensor.size())
print(loaded_tensor.size())
print(loaded_tensor)

# Print the number of unique classes
num_classes = len(label_encoder.classes_)
print("Number of Unique Classes:", num_classes)
