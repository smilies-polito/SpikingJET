import os
import numpy as np

import tonic
from tonic import datasets, transforms

from snntorch import functional as SF
from shd import SHD
import torchsummary as ts

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model_path = '/home/ldapusers/gogebakan/snn_fault_injection/SHD_model/TrainedParameters/state_dict_shd_prova.th'
batch_size		= 256
data_dir		= "./data/shd"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


transform = transforms.Compose(
    [
        transforms.ToFrame(
            sensor_size=tonic.datasets.hsd.SHD.sensor_size,
            n_time_bins=100,
        )
    ]
)

test_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=False,
		transform = transform)

test_loader 	= DataLoader(test_set, batch_size=batch_size, shuffle=True,
		drop_last=True)

net = SHD(device=device)
net.load_state_dict(torch.load(model_path))
net = net.to(device)
# ts.summary(net, input_size=())
print(net)


net.eval()  
correct = 0
total = 0
loss_test_val = 0
for i, (data, targets) in enumerate(iter(test_loader)):
    data = data[:, :, 0, :]
    data = data.to(device)
    # print('data.shape()', data.shape)
    targets = targets.to(device)
    # print('targets.shape()', targets.shape)

    spk_rec = net.forward(data)
    # print('spk_rec.shape()', spk_rec.shape)
    # loss_test_val += loss_fn(spk_rec, targets)
    
    # correct += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
    _,idx = spk_rec.max(1)
    # print("idx", idx.size())
    # print("targets", targets.size())
    correct += np.mean((targets == idx).detach().cpu().numpy())
    total += spk_rec.size(1)


print(f"Test Accuracy: {(correct/total) * 100:.2f}%\n")