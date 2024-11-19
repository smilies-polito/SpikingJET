import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn
import torch.nn.functional as F

class Readout(snn.Leaky):

	def __init__(self, beta):

		super().__init__(beta, reset_mechanism = "none")

	def forward(self, cur, mem):

		_, mem = super().forward(cur, mem)

		return mem, fn.softmax(mem, dim = 1)


class CSNN_AUDIOMNIST(nn.Module):
	def __init__(self, num_inputs, num_hidden = 128,num_outputs = 10, model = "lif", bias = False,beta = 0.9, threshold = 1.):
		super().__init__()
        self.FaultInjector = None
		self.conv1 = nn.Conv1d(1,32,3,1,1)
		#self.conv2=nn.Conv1d(32,64,3,1,1)
		self.pool=nn.MaxPool1d(2)
		self.fc1=nn.Linear(32*20,128)
		self.lif1 = snn.Leaky(beta = beta,threshold = threshold,reset_mechanism	= "subtract")
		#self.fc2 = nn.Linear(in_features = 256,out_features	= 64)
		#self.lif2 = snn.Leaky(beta = beta,threshold = threshold,reset_mechanism	= "subtract")
		self.fc2 = nn.Linear(in_features = 128,out_features	= num_outputs, bias = bias,)
		self.readout = Readout(beta	= beta)
	
	def forward(self, input_spikes):
		mem1 = self.lif1.reset_mem()
		#mem2 = self.lif2.reset_mem()
		mem_out = self.readout.reset_mem()
		mem_out_rec = []
		out_rec = []
		#print(input_spikes.shape)
		for step in range(input_spikes.shape[0]):
			#print(input_spikes[step].shape)
			x_t=self.pool(torch.relu(self.conv1(input_spikes[step])))
			#x_t=self.pool(torch.relu(self.conv2(x_t)))
			#print(x_t.shape)
			x_t = x_t.view(x_t.size(0), -1)
			#print(x_t.shape)
			cur1 = self.fc1(x_t)
			spk1, mem1 = self.lif1(cur1, mem1)
            self.FaultInjector.injection(mem1,spk1, 'lif1', step)
			cur2 = self.fc2(spk1)
			#spk2, mem2 = self.lif2(cur2, mem2)
			#cur3 = self.fc3(spk2)
			mem_out, out = self.readout(cur2, mem_out)
			mem_out_rec.append(mem_out)
			out_rec.append(out)
		return torch.stack(mem_out_rec, dim=0), torch.stack(out_rec, dim=0)
