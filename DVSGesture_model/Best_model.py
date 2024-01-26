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

        #Initialize parameters
        self.beta = torch.tensor([0.5])
        self.gradient = surrogate.fast_sigmoid(slope=25)

        # Initialize layers
        self.conv1 = nn.Conv2d(2, 12, 5)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.gradient, learn_beta=True, learn_threshold=True)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.gradient, learn_beta=True, learn_threshold=True)
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
        x = nn.functional.interpolate(x, size=(2, 32, 32))
        for step in range(x.shape[0]):

          cur1 = F.max_pool2d(self.conv1(x[step]), 2)
          spk1, mem1 = self.lif1(cur1, mem1)

          cur2 = F.max_pool2d(self.conv2(spk1), 2)
          spk2, mem2 = self.lif2(cur2, mem2)

          cur3 = self.fc1(spk2.flatten(1)) # batch x ....
          spk3, mem3 = self.lif3(cur3, mem3)

          spk_rec.append(spk3)
          # mem_rec.append(mem3)

        results =torch.stack(spk_rec)  
        _, idx = results.sum(dim=0).max(1)

        return idx