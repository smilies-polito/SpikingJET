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


model_path = "/home/ldapusers/gogebakan/DVSGesture_model/2024_01_23-01_24_15_PM/weights/CSNN_DVSGesture.th"

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


# Evaluate Test Set Accuracy
correct = 0
total = 0

for i, (data, targets) in enumerate(iter(test_loader)):
  data = nn.functional.interpolate(data, size=(2, 32, 32))
  data = data.to(device)
  targets = targets.to(device)
  spk_rec = net.forward_pass(data)
  correct += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
  total += spk_rec.size(1)

print("Test Set Accuracy: ", (correct/total) * 100, "%")