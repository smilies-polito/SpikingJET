"""Mixed (hybrid) network: a conventional convolutional front end and a spiking back end.

    frame [2, 34, 34] --Conv-BN-ReLU-Pool--Conv-BN-ReLU-Pool-->  (non-spiking, per frame)
                      --Linear--> LIF(128) --Linear--> LI readout (10)     (spiking, over T frames)

Input: N-MNIST frames [B, T, 2, 34, 34] (the NMNIST_CSNN loader). Output: mean readout membrane
over time, [B, 10]. The model needs no fault-injection code: SpikingJET finds the snntorch neurons
(lif1, readout) and the ReLU / Conv / Linear outputs automatically (spikingjet.sites).
"""
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate


class HybridNMNIST(nn.Module):
    def __init__(self, hidden: int = 128, beta: float = 0.9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),     # 17x17
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 8x8
        )
        self.fc1 = nn.Linear(32 * 8 * 8, hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=surrogate.atan())
        self.fc_out = nn.Linear(hidden, 10)
        self.readout = snn.Leaky(beta=beta, learn_beta=True, reset_mechanism="none",
                                 spike_grad=surrogate.atan())
        self.num_steps = None

    def forward(self, x):
        x = x.float()                                            # [B, T, 2, 34, 34]
        B, T = x.shape[:2]
        f = self.features(x.flatten(0, 1)).flatten(1).view(B, T, -1)   # ANN part, all frames at once
        mem1 = self.lif1.init_leaky()
        mem_o = self.readout.init_leaky()
        out = 0.0
        for t in range(T):                                       # SNN part, step by step
            spk1, mem1 = self.lif1(self.fc1(f[:, t]), mem1)
            _, mem_o = self.readout(self.fc_out(spk1), mem_o)
            out = out + mem_o
        return out / T
