import torch
import matplotlib.pyplot as plt
import os

#def mnist(rawfolder, save_path):
# Initialize empty lists to store train and test tensors

rawfolder = "../../data/raw"
save_path = "../../data/processed"


train_images_tensors = []
train_target_tensors = []

# Loop through train files
for i in range(6):
    train_images_file_path = rawfolder+'/train_images_'+str(i)+'.pt'
    train_images_state_dict = torch.load(train_images_file_path)
    train_images_tensors.append(train_images_state_dict)

    train_target_file_path = rawfolder+'/train_target_'+str(i)+'.pt'
    train_target_state_dict = torch.load(train_target_file_path)
    train_target_tensors.append(train_target_state_dict)

# Concatenate tensors along the first dimension (assuming they have the same size)
train_images_tensor = torch.cat(train_images_tensors, dim=0)
train_target_tensor = torch.cat(train_target_tensors, dim=0)

# Normalize the data with mean 0 and standard deviation 1
mean_value = train_images_tensor.mean()
std_value = train_images_tensor.std()
train_images_tensor = (train_images_tensor - mean_value) / std_value

# Save normalized tensors
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.save(train_images_tensor, os.path.join(save_path, 'train_images_tensor.pt'))
torch.save(train_target_tensor, os.path.join(save_path, 'train_target_tensor.pt'))




"""
if __name__ == '__main__':
    # Get the data and process it
    rawfolder = "../../data/raw"
    save_path = "../../data/processed"
    mnist(rawfolder, save_path)
"""
