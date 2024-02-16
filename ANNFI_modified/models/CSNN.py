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

        self.inject = False
        self.xor = True
        self.one = True
        self.current = None
        self.size = {}
        self.mem = {}
        self.faults = {}
        self.mask = {}


    def flatten_layer_dim(self, layer):
      '''
      For a given layer (Multi dimensional array) compute
      the dimension of the flatten array (Mono dimensional array) 
      '''
      n = 1
      for dim in self.size[layer]:
        n = n*dim  
      return n  

    def compute_flatten_index(self, layer, index_arr):
      '''
      Given an index of the layer (Multi dimensional array) return
      the corresponding index of the flatten array
      '''
      index = 0
      for n in range(len(index_arr)):
        i = n+1
        tmp =  index_arr[-i]
        for y in range(n):
          i2 = y+1
          tmp = tmp *self.size[layer][-i2]
        index = index + tmp 
      return index   

    def compute_mask(self):
      '''
      Compute the mask to perform the xor injection
      '''
      for layer, faults in self.faults.items():
        if layer not in self.mask.keys():
          self.mask[layer] = torch.zeros(self.size[layer], dtype=torch.int32)
        for tensor_index, bit_index in faults:
          
          self.mask[layer][tensor_index] += 2**bit_index
          

    def injection_dict(self, layer, tensor_index, bit_index):
        '''
        inject faults using the dictionary
        '''
        
        golden_value = float(self.mem[layer][tensor_index])
        faulty_value = float32_bit_flip(golden_value=golden_value, bit=bit_index)
        self.mem[layer][tensor_index] = faulty_value

    def injection_xor(self, layer):    
      '''
        inject faults using the XOR mask
      '''
      pot_int = self.mem[layer].view(torch.int)
      pot_int_incjeted = torch.bitwise_xor(pot_int, self.mask[layer])
      self.mem[layer] = pot_int_incjeted.view(torch.float)


    def injection(self, layer):
      if self.inject:
        if self.one and self.current[0] == layer:
            self.injection_dict(layer, self.current[1], self.current[2])
        elif self.xor:
          self.injection_xor(layer)
        else:  
          for fault in self.faults[layer]:
            self.injection_dict(layer, fault[0], fault[1])
      else:
        self.size[layer] = self.mem[layer].shape

    def forward(self, x):     

        # Initialize hidden states and outputs at t=0
        self.mem['lif1'] = self.lif1.init_leaky()
        self.mem['lif2'] = self.lif2.init_leaky()
        self.mem['lif3'] = self.lif3.init_leaky()
        
        nu_step = []
        mem_rec = []
        spk_rec = []

        x = nn.functional.interpolate(x, size=(2, 32, 32))
        num_step = 0
        for step in range(x.shape[0]):

          cur1 = F.max_pool2d(self.conv1(x[step]), 2)
          spk1, self.mem['lif1'] = self.lif1(cur1, self.mem['lif1'])
          self.injection('lif1')
          
          cur2 = F.max_pool2d(self.conv2(spk1), 2)
          spk2, self.mem['lif2'] = self.lif2(cur2, self.mem['lif2'])
          self.injection('lif2')

          cur3 = self.fc1(spk2.flatten(1)) # batch x ....
          spk3, self.mem['lif3'] = self.lif3(cur3, self.mem['lif3'])
          self.injection('lif3')
          
          spk_rec.append(spk3)
          num_step+=1

        print(num_step)
        spk_rec =torch.stack(spk_rec)  
        _, results = spk_rec.sum(dim=0).max(1)

        return results
