import tonic
import tonic.transforms as transforms
import os
from torchsummary import summary


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



from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import torch
import torchvision

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./data/cache/nmnist/train')

# no augmentations for the testset
cached_testset = DiskCachedDataset(testset, cache_path='./data/cache/nmnist/test')


trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# neuron and simulation parameters
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.beta1 = torch.ones(20)*0.5
        self.beta2 = torch.ones(10)*0.5

        self.threshold1 = torch.ones(20)
        self.threshold2 = torch.ones(10)
        # self.beta = 0.5
        # self.threshold = 1

        self.spike_grad = surrogate.atan()

        # Initialize layers
        self.fc1 = nn.Linear(17*17*2, 20)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.threshold1, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(20, 10)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.threshold2, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True, output=True)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []
        
        x = x.view(x.shape[0], x.shape[1], -1)
        for step in range(x.shape[0]):
            
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec)


net = Net()
# summary(net, input_size=(2, 34, 34))
# exit()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 2
num_iters = 50


loss_hist = []

acc_hist = []

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(trainloader)):
        data = nn.functional.interpolate(data, size=(2, 17, 17))
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec = net.forward(data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%\n")

        # training loop breaks after 50 iterations
        if i == num_iters:
          break

# Test Set Accuracy
net.eval()  
correct = 0
total = 0
loss_test_val = 0
for i, (data, targets) in enumerate(iter(testloader)):
    data = nn.functional.interpolate(data, size=(2, 17, 17))
    data = data.to(device)
    targets = targets.to(device)

    spk_rec = net.forward(data)
    # loss_test_val += loss_fn(spk_rec, targets)
    correct += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
    total += spk_rec.size(1)


print(f"Test Accuracy: {(correct/total) * 100:.2f}%\n")

path = os.getcwd()
model_weight_PATH = os.path.join(path,"weights")
try:
    os.mkdir(model_weight_PATH)
except:
    pass
model_save_PATH = os.path.join(model_weight_PATH, "NMNIST.th")
torch.save(net.state_dict(), model_save_PATH)


# import matplotlib.pyplot as plt

# # Plot Loss
# fig = plt.figure(facecolor="w")
# plt.plot(acc_hist)
# plt.title("Train Set Accuracy")
# plt.xlabel("Iteration")
# plt.ylabel("Accuracy")
# plt.show()