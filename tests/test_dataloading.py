import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def test_data():
    rawfolder = "data/raw/images"
    labels_file = "data/raw/labels.csv"

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

    # Loop through train files
    for filename in os.listdir(rawfolder):
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
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Convert the list of labels to a tensor
    corresponding_labels_tensor = torch.tensor(corresponding_labels, dtype=torch.long)

    # Concatenate tensors along the first dimension
    train_images_tensor = torch.stack(train_images_tensors, dim=0)

    # Normalize the data with mean 0 and standard deviation 1
    mean_value = train_images_tensor.mean()
    std_value = train_images_tensor.std()
    train_images_tensor = (train_images_tensor - mean_value) / std_value
    
    
    # Assert that the number of loaded images is equal to the number of labels
    assert len(train_images_tensors) == len(corresponding_labels_tensor)

    # Assert that the dimensions of the entire train_images_tensor are as expected
    expected_dimensions = torch.Size([len(train_images_tensors), 3, 224, 224])
    assert train_images_tensor.size() == expected_dimensions

    # Assert that label encoding is working correctly
    for label in corresponding_labels_tensor:
        assert label.item() >= 0  # Labels should be non-negative after encoding

    # Assert that label decoding is working correctly
    decoded_labels = label_encoder.inverse_transform(corresponding_labels_tensor.numpy())
    assert all(label in labels for label in decoded_labels)

    # Assert that the mean and standard deviation of the normalized images are close to 0 and 1, respectively
    assert torch.allclose(train_images_tensor.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(train_images_tensor.std(), torch.tensor(1.0), atol=1e-5)




