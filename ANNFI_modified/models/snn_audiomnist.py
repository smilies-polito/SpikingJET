import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn

class Readout(snn.Leaky):

	def __init__(self, beta):

		super().__init__(beta, reset_mechanism = "none")

	def forward(self, cur, mem):

		_, mem = super().forward(cur, mem)

		return mem, fn.softmax(mem, dim = 1)


class SNN_AUDIOMNIST(nn.Module):
	def __init__(self, num_inputs = 40, num_hidden = 128,num_outputs = 10, model = "lif", bias = False,beta = 0.9, threshold = 1.):
		super().__init__()
        self.FaultInjector = None
		self.fc1 = nn.Linear(in_features	= num_inputs,out_features = num_hidden,bias = bias,)
		self.lif1 = snn.Leaky(beta = beta,threshold = threshold,reset_mechanism	= "subtract")
		self.fc2 = nn.Linear(in_features	= num_hidden,out_features	= num_hidden, bias = bias,)
		self.lif2 = snn.Leaky(beta = beta,threshold = threshold,reset_mechanism	= "subtract")
		self.fc3 = nn.Linear(in_features	= num_hidden,out_features	= num_outputs, bias = bias,)
		self.readout = Readout(beta	= beta)
	
	def forward(self, input_spikes):
		mem1 = self.lif1.reset_mem()
		mem2 = self.lif2.reset_mem()
		mem_out = self.readout.reset_mem()
		mem_out_rec = []
		out_rec = []
		for step in range(input_spikes.shape[0]):
			#print(input_spikes[step].shape)
			cur1 = self.fc1(input_spikes[step])
			spk1, mem1 = self.lif1(cur1, mem1)
            self.FaultInjector.injection(mem1,spk1, 'lif1', step)
			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)
            self.FaultInjector.injection(mem2,spk2, 'lif2', step)
			cur3 = self.fc3(spk2)
			mem_out, out = self.readout(cur3, mem_out)
			mem_out_rec.append(mem_out)
			out_rec.append(out)
		return torch.stack(mem_out_rec, dim=0)
