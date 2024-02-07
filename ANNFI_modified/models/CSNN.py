"""
This is an example model script to understand how to save and load models in torch

Load model with load_model.py
"""

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn
from torch.nn import functional as F
import torch

from FaultGenerators.utils import float32_bit_flip

# Define Network
class CSNN(nn.Module):
    def __init__(self):
        super().__init__() 

        #Initialize parameters
        self.beta1 = torch.ones(14)*0.5
        self.beta2 = torch.ones(5)*0.5
        self.beta3 = torch.ones(11)*0.5

        self.threshold1 = torch.ones(14)
        self.threshold2 = torch.ones(5)
        self.threshold3 = torch.ones(11)
        
        self.gradient = surrogate.fast_sigmoid(slope=25)

        # Initialize layers
        self.conv1 = nn.Conv2d(2, 12, 5)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.threshold1, spike_grad=self.gradient, learn_beta=True, learn_threshold=True)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.threshold2, spike_grad=self.gradient, learn_beta=True, learn_threshold=True)
        self.fc1 = nn.Linear(800, 11)
        self.lif3 = snn.Leaky(beta=self.beta3, threshold=self.threshold3, spike_grad=self.gradient, learn_beta=True, learn_threshold=True, output=True)

        self.mem = {}
        self.inject = False
        self.faults = {}

    def injection(self, layer, tensor_index, bit_index):
        
        golden_value = float(self.mem[layer][tensor_index])
        faulty_value = float32_bit_flip(golden_value=golden_value, bit=bit_index)
        self.mem[layer][tensor_index] = faulty_value

    def forward(self, x):
        '''
        if self.inject:
          for layer, faultss in self.faults.items():
            print(layer)
            for fault in faultss:
              print(fault[0])
              print(fault[1])
        '''      
        # Initialize hidden states and outputs at t=0
        self.mem['lif1'] = self.lif1.init_leaky()
        self.mem['lif2'] = self.lif2.init_leaky()
        self.mem['lif3'] = self.lif3.init_leaky()
        
        '''
        if self.inject:
          for layer, fault_list in self.faults.items():
            for fault in fault_list:
              self.injection(layer, fault[0], fault[1])
        '''      

        mem_rec = []
        spk_rec = []
        #print('x.shape ', x.shape[0])
        x = nn.functional.interpolate(x, size=(2, 32, 32))
        for step in range(x.shape[0]):

          cur1 = F.max_pool2d(self.conv1(x[step]), 2)
          spk1, self.mem['lif1'] = self.lif1(cur1, self.mem['lif1'])
          if self.inject:
            for fault in self.faults['lif1']:
              self.injection('lif1', fault[0], fault[1])
          #print(self.mem['lif1'].shape)
          
          cur2 = F.max_pool2d(self.conv2(spk1), 2)
          spk2, self.mem['lif2'] = self.lif2(cur2, self.mem['lif2'])
          if self.inject:
            for fault in self.faults['lif2']:
              self.injection('lif2', fault[0], fault[1])
          #print(self.mem['lif2'].shape)

          cur3 = self.fc1(spk2.flatten(1)) # batch x ....
          spk3, self.mem['lif3'] = self.lif3(cur3, self.mem['lif3'])
          if self.inject:
            for fault in self.faults['lif3']:
              self.injection('lif3', fault[0], fault[1])
          #print(self.mem['lif3'].shape)
          
          spk_rec.append(spk3)
          # mem_rec.append(mem[2])


        spk_rec =torch.stack(spk_rec)  
        _, results = spk_rec.sum(dim=0).max(1)

        return results
