# imports
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
from torch.nn import functional as F

from datetime import datetime
import os

time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

parent_dir = os.getcwd()
path= os.path.join(parent_dir,'train_logs')
path = os.path.join(parent_dir, time)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Device is: ", device)


dataset = tonic.datasets.DVSGesture(save_to='./data', train=True)
events, label = dataset[0]

# Dataset size
print("Dataset contains ", len(dataset), " samples.")

# Number of events in the first sample
print("There are ", len(events), " events in the first sample.")

# (x-pos, y-pos, polarity, timestamp)
print("A single event: ", events[0])

"""
##############################
# This is not necessary, but is cool to see
# Converting 1000 event blocks into frames with tonic.transforms.ToFrame
sensor_size = tonic.datasets.DVSGesture.sensor_size
frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, event_count=1000)

# Applying the transform to the raw events array
frames = frame_transform(events)

# Shape of the frames object represents (num frames, polarity, width, height)
print(frames.shape)

# Plot some frames
fig, ax = plt.subplots(1, 3)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[0].imshow(frames[0][0])
ax[1].imshow(frames[10][0])
ax[2].imshow(frames[20][0])
###############################
"""

### Transforms
size = tonic.datasets.DVSGesture.sensor_size

# Denoise transform removes outlier events with inactive surrounding pixels for 10ms
denoise_transform = transforms.Denoise(filter_time=10000)

# ToFrame transform bins events into 25 clusters of frames
frame_transform = transforms.ToFrame(sensor_size=size, n_time_bins=25)

# Chain the transforms
all_transform = transforms.Compose([denoise_transform, frame_transform])

# View the transformed data
tf_frames = all_transform(events)
print("Shape of frames object: ", tf_frames.shape)
plt.axis('off')
plt.imshow(tf_frames[0][0])

train_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=True)
test_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=False)
print('Number of elements in Train Set: ', len(train_set))
print('Number of elements in Test Set: ', len(test_set))

### Caching and Dataloaders
cached_trainset = tonic.DiskCachedDataset(train_set, cache_path='./cache/dvsgesture/train')
cached_testset = tonic.DiskCachedDataset(test_set, cache_path='./cache/dvsgesture/test')

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(cached_trainset, batch_size=64, shuffle=True, drop_last=True, collate_fn=tonic.collation.PadTensors(batch_first=False))
test_loader = torch.utils.data.DataLoader(cached_testset, batch_size=32, shuffle=True, drop_last=True, collate_fn=tonic.collation.PadTensors(batch_first=False))

# Visualize shape of sample in dataloader (similar to frames object above, but 2nd element is batch size)
for sample in iter(train_loader):
  print("Shape of sample object: ", sample[0].shape)
  break


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #Initialize parameters
        self.beta = 0.5
        self.gradient = surrogate.fast_sigmoid(slope=25)

        # Initialize layers
        self.conv1 = nn.Conv2d(2, 12, 5)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.gradient, learn_beta=True)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.gradient, learn_beta=True)
        self.fc1 = nn.Linear(800, 11)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.gradient, learn_beta=True, output=True)

    def forward_pass(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        

        mem_rec = []
        spk_rec = []
        # print('x.shape ', x.shape[0])
        for step in range(x.shape[0]):

          cur1 = F.max_pool2d(self.conv1(x[step]), 2)
          spk1, mem1 = self.lif1(cur1, mem1)

          cur2 = F.max_pool2d(self.conv2(spk1), 2)
          spk2, mem2 = self.lif2(cur2, mem2)

          cur3 = self.fc1(spk2.flatten(1)) # batch x ....
          spk3, mem3 = self.lif3(cur3, mem3)

          spk_rec.append(spk3)
          # mem_rec.append(mem3)

        return torch.stack(spk_rec)

net = Net()
print("NET IS: ", net)
net = net.to(device)



# some hyperparameters
optimizer = torch.optim.Adam(net.parameters(), lr=0.002, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

### Training
# 50 epochs; counter just used for test set validation
num_epochs = 50
counter = 0

loss_hist = []
acc_hist = []
test_acc_hist = []

# Training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        # Downsampling image from (128 x 128) to (32 x 32)
        data = nn.functional.interpolate(data, size=(2, 32, 32))
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        # propagating one batch through the network and evaluating loss
        spk_rec = net.forward_pass(data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)

        # print metrics every so often
        if counter % 16 == 0:
          print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
          print(f"Train Accuracy: {acc * 100:.2f}%\n")

          correct = 0
          total = 0

          for i, (data, targets) in enumerate(iter(test_loader)):
            data = nn.functional.interpolate(data, size=(2, 32, 32))
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = net.forward_pass(data)
            correct += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

          test_acc = (correct/total) * 100
          test_acc_hist.append(test_acc)
          print(f"========== Test Set Accuracy: {test_acc:.2f}% ==========\n")

        counter += 1

os.mkdir(path)
model_info_path = os.path.join(path,'model_summary.txt')

with open(model_info_path, 'w') as f:
    f.write(repr(net))
    f.write('\nEpoch number: ' + str(num_epochs))


### Results
test_acc_path = os.path.join(path,'test_acc.png')
fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.savefig(test_acc_path)

train_acc_path = os.path.join(path,'train_acc.png')
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.savefig(train_acc_path)

train_loss_path = os.path.join(path, 'train_loss.png')
fig = plt.figure(facecolor="w")
plt.plot(loss_hist)
plt.title("Test Set Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(train_loss_path)


model_weight_PATH = os.path.join(path,"weights")
os.mkdir(model_weight_PATH)
model_save_PATH = os.path.join(model_weight_PATH, "CSNN_DVSGesture.th")
torch.save(net.state_dict(), model_save_PATH)




# fig, axes = plt.subplots(1, 3, figsize=(18,4))

# # Plot Train Accuracy
# axes[0].plot(acc_hist)
# axes[0].set_title("Train Set Accuracy")
# axes[0].set_xlabel("Iteration")
# axes[0].set_ylabel("Accuracy")

# # Plot Test Accuracy
# axes[1].plot(test_acc_hist)
# axes[1].set_title("Test Set Accuracy")
# axes[1].set_xlabel("Iteration")
# axes[1].set_ylabel("Accuracy")

# # Plot Training Loss
# axes[2].plot(loss_hist)
# axes[2].set_title("Loss History")
# axes[2].set_xlabel("Iteration")
# axes[2].set_ylabel("Loss")

# plt.show()

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
