7.48 KiB
import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn

class CNN_AUDIOMNIST(nn.Module):
	def __init__(self, num_inputs, num_hidden, num_outputs):
		super().__init__()
        self.FaultInjector = None
		self.conv1 = nn.Conv2d(1,32,3,1,1)
		self.pool = nn.MaxPool2d(2,2)
		#self.conv2 = nn.Conv2d(16,32,3,1,1)
		self.fc1 = nn.Linear(32*20*36, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		out_rec=[]
		#print(x.shape)
		#for step in range(x.shape[0]):
		x_t = self.pool(torch.relu(self.conv1(x)))
		#print(x_t.shape)
		x_1 = torch.flatten(x_t, 1)
		x_2 = self.fc1(x_1)
		out = self.fc2(x_2)
		out_rec.append(out)
		return torch.stack(out_rec,dim=0)
