"""Hardware-realistic fault model for SpikingJET.

codec          storage formats, bit-level fault operations, precision profiles
memory_map     memory classes M1-M4, sites, exclusions, hardware-target switch
injector       code faults (M1/M2) and neuron-state / spike-routing faults (M3/M4)
draw           counter-based BER drawing and stratified single-fault lists
metrics        per-fault and per-realization damage metrics
catalog_campaign / ber_campaign   campaign runners used by hw_campaign.py

Hardware (integer-format) counterpart of the FP32 engine in spikingjet/: it works on the
deployment form of SHD_DELAY (models/deployed_shd.py) and has its own memory map and codecs.
"""
