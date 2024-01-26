"""
This is an example model load script to understand how to save and load models in torch.
Also, test dataset imported to check whether model is imported right.

Load from Best_model.py
"""

import tonic
import matplotlib.pyplot as plt
import tonic.transforms as transforms

import torch
import torchvision

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import torch.nn as nn

from datetime import datetime
import os
from Best_model import CSNN
import numpy as np


model_path = "/home/ldapusers/gogebakan/snn_fault_injection/DVSGesture_model/2024_01_26-02_12_04_PM/weights/CSNN_DVSGesture.th"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Device is: ", device)

### Transforms
size = tonic.datasets.DVSGesture.sensor_size

# Denoise transform removes outlier events with inactive surrounding pixels for 10ms
denoise_transform = transforms.Denoise(filter_time=10000)

# ToFrame transform bins events into 25 clusters of frames
frame_transform = transforms.ToFrame(sensor_size=size, n_time_bins=25)

# Chain the transforms
all_transform = transforms.Compose([denoise_transform, frame_transform])
test_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=False)
cached_testset = tonic.DiskCachedDataset(test_set, cache_path='./cache/dvsgesture/test')
test_loader = torch.utils.data.DataLoader(cached_testset, batch_size=32, shuffle=True, drop_last=True, collate_fn=tonic.collation.PadTensors(batch_first=False))

net = CSNN().to(device)



net.load_state_dict(torch.load(model_path))

print(net)

injectable_layer_names = [name.replace('.weight', '') for name, module in net.named_modules()]
print('injectable_layer_names', injectable_layer_names)

print('net.named_parameters', net.named_parameters())
for name, param in net.named_parameters():
  print(name, param.shape)
  print(param)
  print()

# List of the shape of all the layers that contain weight

# net_layer_shape_beta = {name.replace('.beta', ''): param.shape for name, param in net.named_parameters()
#                         if (name.replace('.beta', '') in injectable_layer_names)}

# print('net_layer_shape_beta', net_layer_shape_beta)

# net_layer_shape = {name.replace('.weight', ''): param.shape for name, param in net.named_parameters()
#                         if (name.replace('.weight', '') in injectable_layer_names)}

# print('net_layer_shape', net_layer_shape)

exit()
# List of the injectable params
self.net_layer_params = {name.replace('.weight', ''): param for name, param in self.network.named_parameters()
                          if name.replace('.weight', '') in injectable_layer_names}








# Evaluate Test Set Accuracy
correct = 0
total = 0

for i, (data, targets) in enumerate(iter(test_loader)):
  data = data.to(device)
  targets = targets.to(device)
  outputs = net.forward_pass(data)
  # print()


  # print('targets.shape', targets.shape)
  # print('outputs.shape', outputs.shape)
  correct += np.mean((targets == outputs).detach().cpu().numpy()) * outputs.size(0)
  total += outputs.size(0)

print("Test Set Accuracy: ", (correct/total) * 100, "%")