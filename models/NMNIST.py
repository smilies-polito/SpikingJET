import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn
from torch.nn import functional as F
import torch

# from FaultGenerators.utils import float32_bit_flip

class NMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.FaultInjector = None
        
        self.beta1 = torch.ones(20)*0.5
        self.beta2 = torch.ones(10)*0.5

        self.threshold1 = torch.ones(20)
        self.threshold2 = torch.ones(10)

        self.spike_grad = surrogate.atan()

        # Initialize layers
        self.fc1 = nn.Linear(17*17*2, 20)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.threshold1, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(20, 10)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.threshold2, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True, output=True)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        x = nn.functional.interpolate(x, size=(2, 17, 17))
        x = x.view(x.shape[0], x.shape[1], -1)
        curr_step = 0
        for step in range(x.shape[0]):
            
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            self.FaultInjector.injection(mem1, spk1, 'lif1', curr_step)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            self.FaultInjector.injection(mem2, spk2,'lif2', curr_step)

            spk2_rec.append(spk2)
            curr_step += 1
            # mem2_rec.append(mem2)
        spk2_rec = torch.stack(spk2_rec)
        res_vec = spk2_rec.sum(dim=0)
        res_vec = F.softmax(res_vec)

        return res_vec