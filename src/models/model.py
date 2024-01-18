import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

class ViTModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super(ViTModel, self).__init__()
        
       # Load a pretrained Vision Transformer model
        self.vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)

        # Adjust the output size of the ViT model to match the input size of the first FC layer
        in_features = self.vit_model.head.in_features
        self.vit_model.head = nn.Linear(in_features, 768)  # Assuming the first FC layer expects 768 features

        # fc layers on top of vit
        self.fc_layers = nn.ModuleList()
        for fc_layer in cfg.models.vit.fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, fc_layer.out_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(fc_layer.out_features),
                    nn.Dropout(p=fc_layer.dropout)
                )
            )
            in_features = fc_layer.out_features

        self.output = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Forward pass through the ViT model
        x = self.vit_model(x)

        # Pass through additional fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Final output layer
        x = self.output(x)

        return x


class CustomCNN(nn.Module):
    def __init__(self, cfg, num_classes):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = cfg.models.cnn.input_channels

        self.last_conv_output_channels = None
        for layer_cfg in cfg.models.cnn.conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, layer_cfg.out_channels, kernel_size=layer_cfg.kernel_size, stride=layer_cfg.stride, padding=layer_cfg.padding),
                    nn.ReLU(),
                    nn.BatchNorm2d(layer_cfg.out_channels),
                    nn.MaxPool2d(kernel_size=cfg.models.cnn.pool_size, stride=2),
                    nn.Dropout2d(p=layer_cfg.dropout)
                )
            )
            in_channels = layer_cfg.out_channels
            self.last_conv_output_channels = layer_cfg.out_channels

        # Dynamically calculate the in_features for the first fully connected layer
        self.in_features_fc = self._calculate_conv_output_size(cfg)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.in_features_fc
        for fc_layer in cfg.models.cnn.fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, fc_layer.out_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(fc_layer.out_features),
                    nn.Dropout(p=fc_layer.dropout)
                )
            )
            in_features = fc_layer.out_features

        self.output = nn.Linear(in_features, num_classes)

    def _calculate_conv_output_size(self, cfg):
        image_size = 224  # Example image size (224x224)
        for layer_cfg in cfg.models.cnn.conv_layers:
            image_size = (image_size + 2 * layer_cfg.padding - layer_cfg.kernel_size) // layer_cfg.stride + 1
            image_size = image_size // 2  # Pooling layer
        return image_size * image_size * self.last_conv_output_channels

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        #x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x = self.output(x)
        return x


class MyImprovedCNNModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MyImprovedCNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool1(self.dropout1(self.relu1(self.batchnorm1(self.conv1(x)))))
        x = self.pool2(self.dropout2(self.relu2(self.batchnorm2(self.conv2(x)))))
        x = self.pool3(self.dropout3(self.relu3(self.batchnorm3(self.conv3(x)))))
        x = self.flatten(x)
        x = self.dropout4(self.relu4(self.batchnorm4(self.fc1(x))))
        x = self.dropout5(self.relu5(self.batchnorm5(self.fc2(x))))
        x = self.fc3(x)
        return x