"""SpikingJET injection engine: one engine for spiking, conventional and mixed networks, at
inference (single-fault campaigns, BER sweeps) and during training.

    sites      what can be faulted in a model (parameters, neuron states, activations)
    bitops     float32 bit-flip / stuck-at on tensors
    injector   puts faults into a running model
    faults     single-fault samples (DATE'09), fault-list files, BER maps
    training   faults during training (TrainingFaults)
    campaign   single-fault campaign          ber      BER sweep
    trainer    generic training loop          config   experiment files (TOML)
"""
from .injector import Fault, Injector
from .sites import Site, SiteTable, discover
from .training import TrainingFaults

__all__ = ["Fault", "Injector", "Site", "SiteTable", "discover", "TrainingFaults"]
