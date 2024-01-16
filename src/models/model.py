import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CustomCNN(nn.Module):
    def __init__(self, cfg):
        super(MyImprovedCNNModel, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = cfg.input_channels

        for layer_cfg in cfg.conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, layer_cfg.out_channels, kernel_size=layer_cfg.kernel_size, stride=layer_cfg.stride, padding=layer_cfg.padding),
                    nn.ReLU(),
                    nn.BatchNorm2d(layer_cfg.out_channels),
                    nn.MaxPool2d(kernel_size=cfg.pool_size, stride=2),
                    nn.Dropout2d(p=cfg.dropout_rate)
                )
            )
            in_channels = layer_cfg.out_channels

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = cfg.fc_layers[0].in_features

        for fc_layer in cfg.fc_layers:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, fc_layer.out_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(fc_layer.out_features),
                    nn.Dropout(p=cfg.dropout_rate)
            )
        )
        in_features = fc_layer.out_features
        self.output = nn.Linear(in_features, cfg.output_classes)

def forward(self, x):
    for conv_layer in self.conv_layers:
        x = conv_layer(x)

    x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer

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

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.dropout1(self.relu1(self.batchnorm1(self.conv1(x)))))
        x = self.pool2(self.dropout2(self.relu2(self.batchnorm2(self.conv2(x)))))
        x = self.pool3(self.dropout3(self.relu3(self.batchnorm3(self.conv3(x)))))
        x = self.flatten(x)
        x = self.dropout4(self.relu4(self.batchnorm4(self.fc1(x))))
        x = self.dropout5(self.relu5(self.batchnorm5(self.fc2(x))))
        x = self.fc3(x)
        return x