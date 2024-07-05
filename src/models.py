#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


# class SimpleCNN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
#         self.conv2_drop= self.dropout = nn.Dropout()
#         self.fc1 = nn.Linear(32 * 137, 128)  # 137 là chiều dài của đầu ra sau khi max pooling
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool1d(self.conv1(x), 2))  # Convolutional layer 1
#         x = F.relu(F.max_pool1d(self.conv2(x), 2))  # Convolutional layer 2
#         x = x.view(-1, 32 * 137)  # Flatten the output for fully connected layer
#         x = F.relu(self.fc1(x))   # Fully connected layer 1
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)           # Fully connected layer 2 (output layer)
#         return F.log_softmax(x, dim=1)  

class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.n_chan = input_size
        self.n_classes = num_classes

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=self.n_chan, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.drop = nn.Dropout(p=0.6)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(3968, 100)  # 3968 là kích thước của tensor sau khi max pooling
        self.fc2 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Convolutional layer 1
        x = self.pool(x)           # Max pooling
        x = F.relu(self.conv2(x))  # Convolutional layer 2
        x = self.pool(x)           # Max pooling
        x = x.view(-1, 3968)       # Flatten the output for fully connected layer
        x = F.relu(self.fc1(x))    # Fully connected layer 1
        x = self.drop(x)           # Dropout
        x = self.fc2(x)            # Fully connected layer 2 (output layer)
        return F.log_softmax(x, dim=1)




# class SimpleMLP(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(SimpleMLP, self).__init__()
#         self.dim_in = dim_in
#         self.layer_input = nn.Linear(dim_in, dim_hidden)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.layer_hidden = nn.Linear(dim_hidden, dim_out)

#     def forward(self, x):
#         x = x.view(-1, self.dim_in)
#         x = self.layer_input(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.layer_hidden(x)
#         return F.log_softmax(x, dim=1)



class SimpleMLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SimpleMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # Define layers
        self.layer1 = nn.Linear(dim_in, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, dim_out)
        
        # Define activations and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Flatten the input tensor to (batch_size, dim_in)
        x = x.view(-1, self.dim_in)
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.relu(self.layer5(x))
        x = self.dropout(x)
        x = self.relu(self.layer6(x))
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
