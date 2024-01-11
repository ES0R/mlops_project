import os
import logging
from PIL import Image
import torch
from torchvision import transforms
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
import argparse
import yaml
# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image processing script.')
    parser.add_argument('--dataset', choices=['complete', 'sparse'], default='complete',
                        help='Type of dataset to process (complete or sparse)')
    parser.add_argument('--classes', nargs='*', help='List of class names or indices for the sparse dataset')
    return parser.parse_args()

def read_class_mapping(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_image(image_path, transform):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = transform(img)
        if img_tensor.size(0) == 3:
            return img_tensor
        else:
            logging.warning(f"Skipping image {image_path} due to incorrect number of channels.")
            return None
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

def count_classes_from_mapping(file_path):
    with open(file_path, 'r') as file:
        class_mapping = yaml.safe_load(file)
    return len(class_mapping)


def load_images(root_folder, selected_classes=None, class_mapping=None):
    image_list = []
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    processed_count = 1

    if selected_classes is not None and class_mapping is not None:
        selected_classes = {class_mapping[int(cls)] if cls.isdigit() else cls for cls in selected_classes}

    total_images = 0  # This will be calculated based on relevant subdirectories

    for subdir, dirs, files in os.walk(root_folder):
        # Extract the class name part from the folder name
        label = subdir.split(os.sep)[-1].split('-')[-1]
        if selected_classes is None or label in selected_classes:
            total_images += len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))])

    logging.info(f"Total images to process: {total_images}")

    for subdir, dirs, files in os.walk(root_folder):
        label = subdir.split(os.sep)[-1].split('-')[-1]
        if selected_classes is None or label in selected_classes:
            processed_folder_count = len(selected_classes) if selected_classes else len(class_mapping)+1
            logging.info(f"Processing subdir: {processed_count}/{processed_folder_count} - {label}")
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(subdir, file)
                    img_tensor = load_image(image_path, transform)
                    if img_tensor is not None:
                        image_list.append(img_tensor)
        
            processed_count += 1

    if not image_list:
        raise RuntimeError("No valid images found in the specified folder.")
    return torch.stack(image_list)





def extract_name_from_xml(file_path, selected_classes=None):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for obj in root.findall(".//object"):
            name_element = obj.find("name")
            if name_element is not None and (selected_classes is None or name_element.text in selected_classes):
                return name_element.text
        return None  # Return None if no relevant label is found
    except Exception as e:
        logging.error(f"Error processing XML file {file_path}: {e}")
        return None



def process_folder(folder_path, selected_classes=None):
    res = []
    for folder in os.listdir(folder_path):
        label = folder.split('-')[-1]
        if folder == ".DS_Store" or (selected_classes is not None and label not in selected_classes):
            continue
        xml_files = [file for file in os.listdir(os.path.join(folder_path, folder))]
        for file_name in xml_files:
            file_path = os.path.join(folder_path, folder, file_name)
            name = extract_name_from_xml(file_path, selected_classes)
            if name:
                res.append(name)
    return res



def normalize_tensor(tensor):
    mean_value = tensor.mean()
    std_value = tensor.std()
    return (tensor - mean_value) / std_value

def main():
    try:
        args = parse_arguments()
        class_mapping = read_class_mapping('data/class_mapping.yaml')

        selected_classes = None
        if args.dataset == 'sparse':
            if args.classes:
                selected_classes = set()
                for cls in args.classes:
                    if cls.isdigit() and int(cls) in class_mapping:
                        selected_classes.add(class_mapping[int(cls)])
                    elif cls in class_mapping.values():
                        selected_classes.add(cls)
            else:
                raise ValueError("Sparse dataset selected but no classes specified.")

        root_folder = "data/raw/images/Images"
        folder_path = "data/raw/annotations/Annotation"
        save_path = "data/processed"

        logging.info("Loading images...")
        images_tensor = load_images(root_folder, selected_classes, class_mapping)
        logging.info(f"Loaded {images_tensor.size(0)} images.")

        logging.info("Processing annotations...")
        labels = process_folder(folder_path, selected_classes)
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)
        logging.info(f"Processed {len(labels)} annotations.")

        logging.info("Normalizing images...")
        train_images_tensor = normalize_tensor(images_tensor)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(train_images_tensor, os.path.join(save_path, 'train_images_tensor.pt'))
        torch.save(labels_tensor, os.path.join(save_path, 'train_target_tensor.pt'))
        torch.save(label_encoder, os.path.join(save_path, 'label_encoder.pt'))

        logging.info("Saved processed data.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

