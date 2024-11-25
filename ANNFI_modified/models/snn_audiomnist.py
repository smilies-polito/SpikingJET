import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn

from torch.nn import functional as F

class Readout(snn.Leaky):

    def __init__(self, beta):
        super().__init__(beta, reset_mechanism="none")

    def forward(self, cur, mem):
        _, mem = super().forward(cur, mem)
        return mem, fn.softmax(mem, dim=1)

class SNN_AUDIOMNIST(nn.Module):
	def __init__(self, num_inputs = 40, num_hidden = 150,num_outputs = 10, model = "lif", bias = False,beta = 0.9, threshold = 1.):
		super().__init__()
		self.FaultInjector = None
		self.beta1=torch.ones(num_hidden)*beta
		self.beta2=torch.ones(num_hidden)*beta
		self.beta3=torch.ones(num_outputs)*beta

		self.th1=torch.ones(num_hidden)
		self.th2=torch.ones(num_hidden)
		self.th3=torch.ones(num_outputs)

		self.fc1 = nn.Linear(in_features	= num_inputs,out_features = num_hidden,bias = bias,)
		self.lif1 = snn.Leaky(beta = self.beta1,threshold = self.th1,reset_mechanism="subtract",learn_beta=True,learn_threshold=True)

		self.fc2 = nn.Linear(in_features	= num_hidden,out_features	= num_hidden, bias = bias,)
		self.lif2 = snn.Leaky(beta = self.beta2,threshold = self.th2,reset_mechanism="subtract",learn_beta=True,learn_threshold=True)

		self.fc3 = nn.Linear(in_features	= num_hidden,out_features	= num_outputs, bias = bias,)
		self.lif3 = snn.Leaky(beta=self.beta3, threshold=self.th3, reset_mechanism="none", 
							learn_beta=True, learn_threshold=True)
		#self.readout = Readout(beta	= beta)
	
	def forward(self, input_spikes):
		mem1 = self.lif1.reset_mem()
		mem2 = self.lif2.reset_mem()
		mem3 = self.lif3.reset_mem()
		mem3_rec = []
		spk3_rec = []
		#print(input_spikes.shape[0])
		for step in range(input_spikes.shape[0]):
			#print(input_spikes[step].shape)
			cur1 = self.fc1(input_spikes[step])
			spk1, mem1 = self.lif1(cur1, mem1)
			self.FaultInjector.injection(mem1,spk1, 'lif1', step)

			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)
			self.FaultInjector.injection(mem2,spk2, 'lif2', step)

			cur3 = self.fc3(spk2)
			spk3,mem3=self.lif3(cur3,mem3)
			self.FaultInjector.injection(mem3,spk3, 'lif3', step)

			mem3_rec.append(mem3)
			spk3_rec.append(spk3)
		spk3_rec = torch.stack(spk3_rec, dim=0)
		mem3_rec = torch.stack(mem3_rec, dim=0)
		avg_mem = mem3_rec.mean(dim=0)
		pro = torch.softmax(avg_mem,dim=-1)
		
		return pro
		
		''' This for working with standard flow	
		spk3_rec = torch.stack(spk3_rec)	
		res_vec = spk3_rec.sum(dim=0)
		res_vec = F.softmax(res_vec)
		return res_vec
		'''

