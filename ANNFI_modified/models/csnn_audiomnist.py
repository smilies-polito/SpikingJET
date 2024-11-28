import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


class CSNN_AUDIOMNIST(nn.Module):
    def __init__(self, num_inputs = 40, num_hidden=150, num_outputs=10, model="lif", bias=False, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta1 = torch.ones(num_hidden) * beta
        self.beta2 = torch.ones(num_outputs) * beta
        self.FaultInjector = None
        self.th1 = torch.ones(num_hidden)
        self.th2 = torch.ones(num_outputs)

        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        # self.conv2 = nn.Conv1d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(32 * 20, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="subtract")

        self.fc2 = nn.Linear(
            in_features=num_hidden,
            out_features=num_outputs,
            bias=bias,
        )
        self.lif2 = snn.Leaky(
            beta=self.beta2,
            threshold=self.th2,
            reset_mechanism="none",
            learn_beta=True,
            learn_threshold=True,
        )

    def forward(self, input_spikes):
        input_spikes = input_spikes.permute(1, 0, 2).unsqueeze(2)
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem_out_rec = []
        out_rec = []

        # Iterate through time steps of input spikes
        for step in range(input_spikes.shape[0]):
            x_t = self.pool(torch.relu(self.conv1(input_spikes[step])))
            # x_t = self.pool(torch.relu(self.conv2(x_t)))

            x_t = x_t.view(x_t.size(0), -1)

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            self.FaultInjector.injection(mem1, spk1, 'lif1', step)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            self.FaultInjector.injection(mem2, spk2, 'lif2', step)

            mem_out_rec.append(mem2)
            out_rec.append(spk2)

        mem_out = torch.stack(mem_out_rec, dim=0)
        avg_mem = mem_out.mean(dim=0)
        pro = torch.softmax(avg_mem, dim=-1)

        return pro
