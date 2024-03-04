"""
This is an example model load script to understand how to save and load models in torch.
Also, test dataset imported to check whether model is imported right.

Load from Best_model.py
"""

import tonic
import matplotlib.pyplot as plt
import tonic.transforms as transforms
import torchsummary as ts

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


model_path = "/home/ldapusers/gogebakan/snn_fault_injection/DVSGesture_model/2024_02_06-01_19_28_PM/weights/CSNN_DVSGesture.th"

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
ts.summary(net, input_size=(2, 32, 32))
exit()

# module_classes = (nn.Conv2d, snn.Leaky)
module_classes = (nn.Conv2d)
net.load_state_dict(torch.load(model_path))

print(net)

#It should be working  
# Name of the injectable layers
injectable_layer_names = [name.split('.')[0] for name, module in net.named_modules()
                          if isinstance(module, module_classes)]

# List of the shape of all the layers that contain weight
net_layer_shape = {name: param.shape for name, param in net.named_parameters()
                        if name.split('.')[0] in injectable_layer_names}

# List of the injectable params
net_layer_params = {name: param for name, param in net.named_parameters()
                          if name.split('.')[0] in injectable_layer_names}

print(injectable_layer_names)
print(net_layer_shape)
# print(net_layer_params)


exit()









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