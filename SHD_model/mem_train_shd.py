import os
import numpy as np

import tonic
from tonic import datasets, transforms

import snntorch as snn
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Define Network
class Net(nn.Module):
	def __init__(self, 	device):

		super().__init__()

		
		self.sigmoid_slope		= 100

		self.tau_mem			= 10e-3
		self.tau_syn			= 5e-3

		#Network dimensions
		self.num_inputs = tonic.datasets.hsd.SHD.sensor_size[0]
		self.num_hidden = 200
		self.num_outputs = 20
		self.time_step = 0.001

		self.alpha1 = torch.ones(self.num_hidden)*(np.exp(-self.time_step/self.tau_syn))
		self.beta1 = torch.ones(self.num_hidden)*(np.exp(-self.time_step/self.tau_mem))
		self.threshold1 = torch.ones(self.num_hidden)

		self.alpha2 = torch.ones(self.num_outputs)*(np.exp(-self.time_step/self.tau_syn))
		self.beta2 = torch.ones(self.num_outputs)*(np.exp(-self.time_step/self.tau_mem))
		self.threshold2 = torch.ones(self.num_outputs)

		# Fast sigmoid surrogate gradient
		self.spike_grad = surrogate.fast_sigmoid(slope=self.sigmoid_slope)
		

		# Initialize layers
		self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
		self.fb1 = nn.Linear(self.num_hidden, self.num_hidden)

		self.lif1 = snn.Synaptic(alpha=self.alpha1, beta=self.beta1, threshold=self.threshold1,
						   learn_alpha=False, learn_beta=False, learn_threshold=False,
						   spike_grad = self.spike_grad)

		self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)

		self.lif2 = snn.Synaptic(alpha=self.alpha2, beta=self.beta2, threshold=self.threshold2,
						   	learn_alpha=False, learn_beta=False, learn_threshold=False,
							spike_grad = self.spike_grad)

	def forward(self, input_spikes):
		
		# Initialize hidden states at t=0
		syn1, mem1 = self.lif1.init_synaptic()
		syn2, mem2 = self.lif2.init_synaptic()

		# Record the final layer
		spk2_rec = []
		mem2_rec = []

		input_spikes = input_spikes.float()
		spk1 = torch.zeros(self.num_hidden).to(device)


		for step in range(input_spikes.shape[1]):
			cur1 = self.fc1(input_spikes[:, step, :]) + self.fb1(spk1)
			spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
			cur2 = self.fc2(spk1)
			spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

			spk2_rec.append(spk2)
			mem2_rec.append(mem2)

		return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def print_batch_accuracy(net, data, targets, train=False):

	spk_rec, mem_rec= net(data[:, :, 0, :])
	# print("spk_rec.size()", spk_rec.size())
	# print("mem_rec.size()", mem_rec.size())
	# input()

	spk_stack = torch.sum(spk_rec, dim=0)
	_, am = torch.max(spk_stack, dim=1)

	acc = np.mean((targets == am).detach().cpu().numpy())

	if train:
		print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
	else:
		print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
	return acc


def train_printer(net,epoch, iter_counter, loss_hist,
		test_loss_hist, counter, data, targets, test_data,
		test_targets):

	print(f"Epoch {epoch}, Iteration {iter_counter}")

	print(f"Train Set Loss: {loss_hist[counter]:.2f}")

	print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")

	train_acc = print_batch_accuracy(net, data, targets, train=True)

	test_acc = print_batch_accuracy(net, test_data, test_targets, train=False)

	print("\n")
	return train_acc, test_acc


n_epochs		= 150
n_steps			= 100

batch_size		= 256


sigmoid_slope		= 100

data_dir		= "./data/shd"
trained_param_dir	= "./TrainedParameters"

if not os.path.exists(trained_param_dir):
	os.makedirs(trained_param_dir)

trained_param_dict	= trained_param_dir + "/state_dict_shd_advanced.th"

# Optimizer
adam_beta1		= 0.9
adam_beta2		= 0.999
lr			= 1e-4
weight_decay		= 1e-5

transform = transforms.Compose(
    [
        transforms.ToFrame(
            sensor_size=tonic.datasets.hsd.SHD.sensor_size,
            n_time_bins=100,
        )
    ]
)

train_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=True, transform
		= transform)
test_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=False,
		transform = transform)

train_loader 	= DataLoader(train_set, batch_size=batch_size, shuffle=True,
		drop_last=True)
test_loader 	= DataLoader(test_set, batch_size=batch_size, shuffle=True,
		drop_last=True)


net = Net(device=device)


log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas = (adam_beta1,
		adam_beta2), weight_decay = weight_decay)

loss_hist = []
test_loss_hist = []
train_acc_hist = []
test_acc_hist = []
iter_hist = []
counter = 0

net.to(device)

print("Training has started")
# Outer training loop
for epoch in range(n_epochs):

	iter_counter = 0
	train_batches	= iter(train_loader)

	# Minibatch training loop
	for data, labels in train_batches:

		inputs = data.to(device)
		y = labels.to(device)

		# Forward pass
		net.train()

		spk_rec, mem_rec = net(inputs[:, :, 0, :])
		m, _ = torch.max(mem_rec, 0)
		log_p_y = log_softmax_fn(m)

		loss_val = loss_fn(log_p_y, y)

		# Gradient calculation + weight update
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		# Store loss history for future plotting
		loss_hist.append(loss_val.item())

		# Test set
		with torch.no_grad():

			net.eval()

			test_data, test_labels = next(iter(test_loader))

			test_inputs = test_data.to(device)
			test_y = test_labels.to(device)

			# Test set forward pass
			test_spk_rec, test_mem_rec = net(test_inputs[:, :, 0, :])

			test_m, _ = torch.max(test_mem_rec, 0)
			test_log_p_y = log_softmax_fn(test_m)

			test_loss = loss_fn(test_log_p_y, test_y)

			# Test set loss
			test_loss_hist.append(test_loss.item())

			# Print train/test loss/accuracy
			if counter % 20 == 0 and counter > 0:
				train_acc, test_acc = train_printer(net, epoch,
						iter_counter, loss_hist,
						test_loss_hist, counter, inputs,
						y, test_inputs,
						test_y)
				train_acc_hist.append(train_acc)
				test_acc_hist.append(test_acc)
				iter_hist.append(counter)
			counter += 1
			iter_counter +=1

# Save parameters for inference
torch.save(net.state_dict(), trained_param_dict)

plt.plot(iter_hist, train_acc_hist)
plt.plot(iter_hist, test_acc_hist)

plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("Accuracy_plot.png")



