import tonic
import numpy as np

import snntorch as snn
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.nn import functional as F


# Define Network
class SHD(nn.Module):
	def __init__(self, 	device):

		super().__init__()
		self.FaultInjector = None
		
		self.sigmoid_slope		= 100
		self.device = device
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
						   learn_alpha=True, learn_beta=True, learn_threshold=True,
						   spike_grad = self.spike_grad)

		self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)

		self.lif2 = snn.Synaptic(alpha=self.alpha2, beta=self.beta2, threshold=self.threshold2,
						   	learn_alpha=True, learn_beta=True, learn_threshold=True,
							spike_grad = self.spike_grad)

	def forward(self, input_spikes):
		input_spikes = input_spikes[:, :, 0, :]
		# Initialize hidden states at t=0
		syn1, mem1 = self.lif1.init_synaptic()
		syn2, mem2 = self.lif2.init_synaptic()

		# Record the final layer
		# spk2_rec = []
		mem2_rec = []

		input_spikes = input_spikes.float()
		spk1 = torch.zeros(self.num_hidden).to(self.device)

		curr_step = 0
		for step in range(input_spikes.shape[1]):

			cur1 = self.fc1(input_spikes[:, step, :]) + self.fb1(spk1)
			spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
			self.FaultInjector.injection(mem1,spk1, 'lif1', curr_step)

			cur2 = self.fc2(spk1)
			spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
			self.FaultInjector.injection(mem2, spk2, 'lif2', curr_step)

			# spk2_rec.append(spk2)
			mem2_rec.append(mem2)
			curr_step += 1

		# spk2_rec = torch.stack(spk2_rec, dim=0)
		# spk_stack = torch.sum(spk2_rec, dim=0)

		mem2_stack = torch.stack(mem2_rec, dim=0)
		mem_stack, _ = torch.max(mem2_stack,dim=0)

		# spk_stack = F.softmax(spk_stack,dim=1)
		mem_stack = F.softmax(mem_stack,dim=1)

		return mem_stack