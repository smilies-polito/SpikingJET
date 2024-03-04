import tonic
import tonic.transforms as transforms
import torchsummary as ts

from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import torch
import torchvision
from nmnist import NMNIST 
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

model_path = '/home/ldapusers/gogebakan/snn_fault_injection/N-MNIST_model/weights/NMNIST.th'
batch_size = 128


sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         n_time_bins=100)
                                     ])

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)





transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./data/cache/nmnist/train')

# no augmentations for the testset
cached_testset = DiskCachedDataset(testset, cache_path='./data/cache/nmnist/test')


trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

net = NMNIST()
net.load_state_dict(torch.load(model_path))

net = net.to(device)
ts.summary(net, input_size=(2, 17, 17))
exit()
# Test Set Accuracy
net.eval()  
correct = 0
total = 0
loss_test_val = 0
for i, (data, targets) in enumerate(iter(testloader)):
    data = data.to(device)
    targets = targets.to(device)

    spk_rec = net.forward(data)
    # loss_test_val += loss_fn(spk_rec, targets)
    correct += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
    total += spk_rec.size(1)


print(f"Test Accuracy: {(correct/total) * 100:.2f}%\n")