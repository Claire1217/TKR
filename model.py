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
