import torchvision.models as models
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from tkr_generator import *
import os

import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "nibabel"])

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class channel1(nn.Module):
	def __init__(self):
		super(channel1, self).__init__()
		self.conv1 = nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(96, 10, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(10, 4, kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(4096, 120)
		self.fc2 = nn.Linear(120, 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool(self.relu1(x))
		
		x = self.conv2(x)
		x = self.pool(self.relu1(x))

		x = self.conv3(x)
		x = self.pool(self.relu1(x))


		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.fc2(x)

		return x

def train(num_epochs, model_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    if model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
                # Download torchvision pretrained model from: https://download.pytorch.org/models/resnet34-333f7ec4.pth
        model.load_state_dict(torch.load('tkr/resnet34-333f7ec4.pth'))
        model.fc = nn.Linear(model.fc.in_features, 1, bias=True)
        model.to(device)
    elif model_type == 'densenet':
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1, bias=True)
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    elif model_type == 'cnn':
        model = channel1().float()

    if torch.cuda.is_available():
        model.cuda()
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=Param['LR'], weight_decay=Param['WC'])
    best_accuracy = 0.0

    print(os.getcwd())
    train_loader = TKRDataset('tkr/tkr_data', Param['PL'])
    

    # Define your execution device

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
    for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs


            if model_type == 'resnet34' or model_type == 'densenet':
                images = np.stack((images,)*3, axis=1)
                images = torch.tensor(images, dtype=float).to(device, dtype=float)
                labels = torch.tensor(labels, dtype=float).to(device)
                optimizer.zero_grad()
            # predict classes using images from the training set
                outputs = model(images.reshape(4,3,256,256).float())
                pred = torch.sigmoid(outputs)
            else:
            # zero the parameter gradients
                images = torch.tensor(images, dtype=float).to(device, dtype=float)
                labels = torch.tensor(labels, dtype=float).to(device)
                optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = model(images.reshape(4,1,256,256).float())
                pred = torch.sigmoid(outputs)
            
            # compute the loss based on model output and real labels
            loss = loss_fn(pred[:,0], labels.float())
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
             # extract the loss value
            if i % 20 == 0:     
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                # zero the loss
                if running_loss / 20 < 0.2:
                    los = running_loss /20
                    if Param['PL'] == 1:
                        torch.save(model.state_dict(), f'tkr/weights/{model_type}_coronal_weights_iter{i}_loss{los}.pth')
                    else:
                        torch.save(model.state_dict(), f'tkr/weights/{model_type}_saggital_weights_iter{i}_loss{los}.pth')
                
                    
                running_loss = 0.0 

Param = {
    'LR': 0.0001,
    'WC': 0.0001,
    'PL': 0
}
# train(1, 'cnn')
# train(1, 'densenet')
# train(1, 'resnet34')


# Param = {
#     'LR': 0.0001,
#     'WC': 0.0001,
#     'PL': 1
# }

train(1, 'densenet')
# # train(1, 'resnet34')
# train(1, 'cnn')