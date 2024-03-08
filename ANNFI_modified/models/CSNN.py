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

# Define Network
class CSNN(nn.Module):
    def __init__(self):
        super().__init__() 

        self.FaultInjector = None

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



    def forward(self, x):     

        # Initialize hidden states and outputs at t=0
        mem_lif1 = self.lif1.init_leaky()
        mem_lif2 = self.lif2.init_leaky()
        mem_lif3 = self.lif3.init_leaky()
        
        nu_step = []
        mem_rec = []
        spk_rec = []

        x = nn.functional.interpolate(x, size=(2, 32, 32))
        curr_step = 0
        for step in range(x.shape[0]):

          cur1 = F.max_pool2d(self.conv1(x[step]), 2)
          spk1, mem_lif1 = self.lif1(cur1, mem_lif1)
          self.FaultInjector.injection(mem_lif1, spk1, 'lif1', curr_step)
      
          cur2 = F.max_pool2d(self.conv2(spk1), 2)
          spk2, mem_lif2 = self.lif2(cur2, mem_lif2)
          self.FaultInjector.injection(mem_lif2, spk2, 'lif2', curr_step)
          

          cur3 = self.fc1(spk2.flatten(1)) # batch x ....
          spk3, mem_lif3 = self.lif3(cur3, mem_lif3)
          self.FaultInjector.injection(mem_lif3, spk3, 'lif3', curr_step)
          
          spk_rec.append(spk3)
          curr_step+=1

        
        spk_rec =torch.stack(spk_rec)  

        res_vec = spk_rec.sum(dim=0)

        res_vec = F.softmax(res_vec)
        return res_vec
