# SpikingJET

Fault injection for spiking (SNN), conventional (CNN / DNN) and mixed neural networks in PyTorch,
at inference and during training.

| Tool | What it measures | Command |
|---|---|---|
| **Single-fault campaign** | one fault at a time (bit-flip or stuck-at) in weights, neuron states (membrane potential, spikes) and activations; statistical sample (DATE'09) | `spikingjet.py campaign` (or `main.py`) |
| **BER sweep** | many random faults at once at a bit error rate, in weights, neuron states and activations; accuracy vs BER | `spikingjet.py ber` (or `ber_eval.py`) → `ber_report.py` |
| **Faults during training** | fault-aware training (robustness), and faults in the training hardware: weights, neuron states / activations, gradients, optimizer state | `spikingjet.py train`, or `--faults` on the training scripts |
| **Hardware fault model** | faults in the memories of a neuromorphic chip (quantized weights, delays, neuron parameters, membrane state, spikes) | `deploy_shd.py` → `hw_campaign.py` |

One engine ([`spikingjet/`](spikingjet)) serves every tool and every model type: it finds what can
be faulted in any `nn.Module` by itself. The first three tools work on every network registered in
[`networks.py`](networks.py); the hardware fault model works on `SHD_DELAY`.

## Quick start

```bash
singularity build --fakeroot faulty_snn.sif faulty_snn.def      # once (see Setup)
singularity shell --nv faulty_snn.sif                           # from the repo root
python selftest.py                                              # does everything work? (~3 min)
python download_datasets.py shd                                 # data of SHD_DELAY
python spikingjet.py sites -n SHD_DELAY                         # what can be faulted
python spikingjet.py campaign -c experiments/campaign_snn.toml  # single-fault campaign
python spikingjet.py ber -n SHD_DELAY --set out=ber_results/shd.csv && python ber_report.py ber_results/shd.csv
python spikingjet.py train -c experiments/train_fault_aware.toml   # training with faults (needs --train data)
```

## Contents

```
spikingjet.py            command line: sites | campaign | ber | train | config
spikingjet/              the engine
  sites.py                 finds the fault sites of any model
  injector.py              puts faults into a running model (inference and training)
  bitops.py, faults.py     float32 bit faults, fault lists (DATE'09), BER maps
  training.py, trainer.py  faults during training; generic trainer
  campaign.py, ber.py      single-fault campaign, BER sweep
  config.py                experiment files (TOML) and their defaults
experiments/             example experiment files
networks.py              registry of networks and datasets  <- the only file to edit for a new network
main.py, ber_eval.py     single-fault campaign / weight BER sweep with their original options
analyse.py, pt_to_csv.py campaign .pt outputs -> CSV;  ber_report.py: BER tables and plots
train_snn.py, train_shd_delays.py, train_cnn.py   training recipes (all accept --faults)
deploy_shd.py, hw_campaign.py, hwfault/           hardware fault model (quantized deployment form)
verify_snn.py            check a network before a campaign
download_datasets.py     download the datasets into ./data
selftest.py              check that every tool works
models/                  architectures; models/pretrained_models/ checkpoints
faulty_snn.def           Singularity / Apptainer container
tests/                   pytest
```

## Setup

Container (recommended). [`faulty_snn.def`](faulty_snn.def) holds everything the code needs,
DCLS (training of SHD_DELAY) included:

```bash
singularity build --fakeroot faulty_snn.sif faulty_snn.def   # or: sudo singularity build ...
singularity test  --nv faulty_snn.sif                        # checks the packages and the GPU
singularity shell --nv faulty_snn.sif                        # then run the commands below from the repo root
```

The container ignores the packages in your `~/.local`, so it always runs the versions pinned in
the `.def`. Instead of `shell` you can prefix single commands with
`singularity exec --nv faulty_snn.sif`.

Or with pip (Python 3.10): `pip install -r requirements.txt` (see the file for the CUDA build of
PyTorch and for DCLS).

**Check the installation:**

```bash
python selftest.py            # ~3 min, no download: tests and smoke runs on synthetic data
python selftest.py --full     # every tool on every included network (downloads the datasets; ~20 min plus download time)
```

It prints a PASS / FAIL table; each step's log is in `output/selftest/`.

### Datasets

```bash
python download_datasets.py            # test sets of every dataset (enough for all campaigns)
python download_datasets.py --train    # also the training sets (needed to train)
python download_datasets.py shd        # one dataset: shd, nmnist, dvsgesture, cifar10
python download_datasets.py --list     # what is already in ./data
```

Each file comes from its official source and is checked against its MD5. Running the
script first is optional: the tools download what they miss on first use. DVSGesture is fetched
through the figshare API, because figshare refuses tonic's direct links from scripts.

| Dataset | Networks | Download (test / train) |
|---|---|---|
| SHD | SHD_DELAY, SHD_RLIF | 38 MB / 131 MB (train always fetched: `deploy_shd.py` calibrates on it) |
| N-MNIST | NMNIST, NMNIST_CSNN, NMNIST_HYBRID | 170 MB / 1 GB |
| DVSGesture | CSNN | 0.6 GB / 2.3 GB |
| CIFAR-10 | ResNet*, VGG9 | 170 MB (both splits) |

**Numerics:** every campaign runs in exact FP32. On Ampere and newer GPUs PyTorch would otherwise
use TF32 in convolutions, which hides bit-flips in the low mantissa bits. `tf32 = true` in an
experiment file (or `--tf32`) turns it back on for speed.

## Networks included

| Name | Type | Dataset | Checkpoint (clean test accuracy) |
|---|---|---|---|
| `SHD_DELAY` | SNN with learnable delays, 2 x 256 LIF | SHD | included (94.26 %) |
| `NMNIST_CSNN` | convolutional SNN | N-MNIST | included (99.08 %) |
| `NMNIST` | SNN (snntorch) | N-MNIST | included (81.39 %) |
| `CSNN` | convolutional SNN (snntorch) | DVSGesture | included (80.47 %) |
| `NMNIST_HYBRID` | mixed: conv + ReLU front end, LIF back end | N-MNIST | included (99.06 %) |
| `SHD_RLIF` | recurrent SNN | SHD | train with `train_snn.py` |
| `ResNet20/32/44/56`, `ResNet18`, `VGG9` | CNN | CIFAR-10 | train with `train_cnn.py` or `spikingjet.py train` |

## Fault sites, fault models, experiment files

**Fault sites** are found automatically (`python spikingjet.py sites -n NETWORK` lists them):

| Target | Sites | Found how |
|---|---|---|
| `params` | weights, biases, neuron parameters (`fc1.weight`, `lif1.beta`) | `named_parameters()`; `sites.param_scope = "spikingjet"` keeps the layers with Conv / Linear / LIF (default), `"all"` keeps everything |
| `neurons` | membrane potential and spikes of spiking layers at every timestep (`lif1.potential`, `lif1.spike`) | forward hooks on every snntorch neuron, or the model's own `FaultInjector.injection` call (below) |
| `activations` | output of any module of `sites.activation_types` (ReLU, Conv, Linear, ...: `features.2.output`) | forward hooks |
| `grads`, `optimizer` | gradients and optimizer state (momentum, Adam moments) of the parameters | training only |

`select` / `exclude` pick sites with glob patterns, e.g. `select = ["*.potential"]`,
`exclude = ["fc_out.*"]`.

**Fault models:** `bit-flip` or `stuck-at` (0 / 1) of one stored bit of a float32 value.
`bits` restricts them: `all`, `mantissa`, `exponent`, `sign`, `low-mantissa`, a range `[0, 22]`,
or a list `[1, 5, 30]`.

**Persistence:**

| | `transient` | `upset` | `permanent` |
|---|---|---|---|
| params | present for one step / inference, then removed | bit flipped in memory, stays until rewritten | stuck cell, re-applied after every update |
| neurons, activations | fresh faults at every timestep / call (campaign: one random timestep) | - | same positions at every timestep / call |
| grads | in the gradients of one step | same as transient | fixed positions, every step |
| optimizer | - | flipped in the stored state | stuck cells, every step |

**Experiment files** (TOML, examples in [`experiments/`](experiments)) list only what differs
from the defaults; `python spikingjet.py config` prints all of them with comments in
[`spikingjet/config.py`](spikingjet/config.py). `-n` sets the network and `--set KEY=VALUE`
changes any key; an unknown key is an error with a suggestion.

```bash
python spikingjet.py campaign -c experiments/campaign_snn.toml --set campaign.max_faults=200 --set batch_size=128
python spikingjet.py train -c experiments/train_fault_aware.toml --set "train.faults[0].ber=1e-4"
python spikingjet.py ber -n NMNIST_HYBRID --set checkpoint=models/pretrained_models/NMNIST_HYBRID_fat.th
```

`checkpoint` loads other weights than the registered ones, e.g. a fault-aware-trained model.

## 1. Single-fault campaign

```bash
python spikingjet.py campaign -c experiments/campaign_snn.toml          # SNN: weights + neuron states
python spikingjet.py campaign -c experiments/campaign_activations.toml  # mixed net: activations + neurons
python main.py -n SHD_DELAY --use-cuda --fault-mode bit-flip            # the original command line
python analyse.py -n SHD_DELAY -b 64 -fl 51196_fault_list.csv           # optional: recompute the per-layer table
python pt_to_csv.py -n SHD_DELAY -b 64                                  # optional: raw outputs as CSV
```

- The fault list is a statistical sample (DATE'09 formula: `error_margin` 3 %, `confidence`
  99 %) of all (site element, bit) pairs of the chosen `targets` (default `params` + `neurons`).
  `campaign.seed` gives an independent sample; the same seed always gives the same list.
- Every fault runs on the whole test set. Weight faults stay for the whole inference; neuron and
  activation faults hit every timestep (`neuron_persistence = "permanent"`) or one random timestep
  (`"transient"`).
- Results:
  - `output/fault_list/<network>/<seed>_fault_list.csv`: the faults;
  - `.../<seed>_fault_list_result_batch_size_<b>.csv`: per site, the fraction of fully masked
    faults, critical faults (`SDC_1`, prediction changed), output-score deviations
    (`SDC_0_5%` ... `SDC_20`), and the faulty / clean accuracy (`facc`, `acc`);
  - `output/<network>/pt/`: every output score (`save_outputs = false` skips them; about 2 GB
    for SHD_DELAY), used by `analyse.py` and `pt_to_csv.py` (`-b` = the campaign batch size).
- `max_faults` caps the sample for a quick check (the DATE'09 margin then no longer holds);
  `fault_list` reruns an existing list.

## 2. BER sweep (many faults at once)

```bash
python spikingjet.py ber -c experiments/ber_mixed.toml      # weights + membrane potentials together
python ber_eval.py --network SHD_DELAY NMNIST_CSNN --bers 1e-7 1e-6 1e-5 1e-4 \
    --bits all mantissa --realizations 10 --out ber_results/snn.csv   # weights only, several networks
python ber_report.py ber_results/snn.csv --plot ber_results/snn.png
```

- Every selected bit fails independently with probability BER; each realization runs the whole
  test set. `[[ber.faults]]` groups choose targets, sites, bits and fault model; the swept BER
  applies to all of them (no group = weight bit-flips).
- Weight fault maps depend only on (seed, BER, bits, realization), so two networks with the same
  shapes see the same faults (paired comparison).
- `ber_eval.py` (weights only, original options): `--bits`, `--mode stuck-at`, `--scope all`,
  `--synthetic` (random inputs, no dataset). `ber_report.py` prints accuracy vs BER and the
  tolerable BER at a 1 pp loss (`--loss`); `--ref <network>` adds paired differences.

## 3. Faults during training

Two uses with the same mechanism:

- **Fault-aware training** (robustness): transient faults in the forward / backward pass, the
  update applied to clean weights; afterwards `eval_bers` measures the accuracy under weight
  bit-flips. [`experiments/train_fault_aware.toml`](experiments/train_fault_aware.toml)
- **Training resilience**: faults in the memories of the training hardware (stuck weight cells,
  upsets, faulty gradients, optimizer state). [`experiments/train_resilience.toml`](experiments/train_resilience.toml)

```bash
python download_datasets.py nmnist --train
python spikingjet.py train -c experiments/train_fault_aware.toml
python spikingjet.py train -c experiments/train_resilience.toml --synthetic --set train.epochs=1   # smoke test
python train_snn.py --network NMNIST_CSNN --faults experiments/train_fault_aware.toml           # own recipes
```

A group (`[[train.faults]]`) also has a schedule: epochs `[start_epoch, stop_epoch)`, every
`every`-th step, with probability `prob`. `train.init = "checkpoint"` fine-tunes the trained
weights instead of starting from scratch. The log (`<out>_train_log.csv`) has the number of faults
per group and epoch; the checkpoint stores the config, the totals and the robustness results.

To add faults to your own training loop:

```python
from spikingjet.training import TrainingFaults
from spikingjet.config import load_config

tf = TrainingFaults(load_config("experiments/train_fault_aware.toml"), model, optimizer, example_batch)
for epoch in range(epochs):
    for x, y in loader:
        tf.before_forward(epoch)
        loss = criterion(model(x), y)
        optimizer.zero_grad(); loss.backward()
        tf.after_backward()          # transient weight faults removed, gradient faults
        optimizer.step()
        tf.after_step()              # stuck cells re-applied, optimizer-state faults
tf.close()
```

## 4. Hardware fault model (neuromorphic memories)

Supported network: `SHD_DELAY`. The network is first converted to its deployment form: integer
codes in the formats of a precision profile. Faults then hit the memories of the chip:

| Class | Memory | Contents |
|---|---|---|
| M1 | synapse memory | weights, delays |
| M2 | neuron configuration | bias, threshold, leak |
| M3 | neuron state | membrane potential |
| M4 | spike routing | spikes |

```bash
python deploy_shd.py --profile default --out models/pretrained_models/SHD_DELAY_deployed.th
python hw_campaign.py map --checkpoint models/pretrained_models/SHD_DELAY_deployed.th   # memory map
```

**BER sweep:** write a config such as `ber.json`:

```json
{"base_seed": 1, "bers": [1e-5, 1e-4, 1e-3, 1e-2], "realizations": 20,
 "sweeps": ["joint", "M1", "M2", "M3", "M4"]}
```

```bash
python hw_campaign.py ber --checkpoint models/pretrained_models/SHD_DELAY_deployed.th \
    --config ber.json --out hw_results/ber
```

- `joint` puts faults in every class at once; `M1` ... `M4` put them in one class only.
- Optional config keys: `"rates": {"M1": 1e-4}` (fixed rate per class), `"split"`, `"batch_size"`.
- Output: `hw_results/ber/realizations.csv`, one row per (sweep, BER, realization) with accuracy,
  accuracy loss, confidence loss and the number of faults per class. Resumable.

**Single-fault catalog:**

```bash
python hw_campaign.py faults  --checkpoint models/pretrained_models/SHD_DELAY_deployed.th \
    --fraction 0.01 --out hw_results/faults.csv.gz
python hw_campaign.py catalog --checkpoint models/pretrained_models/SHD_DELAY_deployed.th \
    --faults hw_results/faults.csv.gz --out hw_results/catalog
```

- The catalog measures each fault separately. It is resumable: run it again to continue.
- With the defaults (`--replicas 64 --batch-size 256`) it needs about 13 GB of GPU memory; lower
  `--replicas` on a smaller or shared GPU.
- Output: `hw_results/catalog/records.csv`, one row per fault.

**Options:**
- `--persistence`: `permanent`, `transient`, `mixed` (default), or JSON per class, e.g.
  `'{"M1": "permanent", "M3": "transient"}'`.
- Precision profiles live in `hwfault/profiles/*.toml`: `default` (int8 weights, 16-bit fixed-point
  membrane), `int4w`, `fp32`. Copy one to define your own and pass `--profile <file>`.

## Train a network

```bash
python download_datasets.py --train                          # training sets (once)
python train_shd_delays.py                                   # SHD_DELAY (150 epochs; DCLS, included in the container)
python train_snn.py --network NMNIST_CSNN                    # or SHD_RLIF
python train_cnn.py -n ResNet20 --epochs 200                 # any CNN of networks.py (CIFAR-10)
python spikingjet.py train -c experiments/train_hybrid.toml  # any network, generic trainer (config file)
python verify_snn.py --network ResNet20                      # clean accuracy + injection check
```

- Checkpoints are written to the path registered in `networks.py`, where every tool finds them
  (`--out` / `out = ...` writes elsewhere).
- All of them accept faults during training (`--faults FILE`, or `[[train.faults]]`).
- `--synthetic` checks a training pipeline on random data in seconds.

## Add your own network (SNN, CNN, mixed or any PyTorch model)

1. Put the model in `models/my_net.py`. `forward(x)` must return one score vector per sample,
   shape `[batch, classes]`.
2. Register it in `networks.py`:

   ```python
   "MyNet": Net(_cls("my_net", "MyNet"), f"{PRE}/MyNet.th", "cifar10",
                Example((3, 32, 32), 10, spikes=False)),
   ```

   - The arguments are: builder, checkpoint, dataset, example.
   - `Example(shape, classes)` gives the shape of one input sample (without the batch) and the
     number of classes. Add `time_first=True` if `forward` takes `[T, B, ...]`, and `spikes=False`
     for real-valued inputs such as images. The tests, `--synthetic` runs and `spikingjet.py sites`
     use it to run the network on random data, before any dataset is downloaded.
   - The checkpoint is a `state_dict`, or `{"state_dict": ...}`; it must match the model exactly.
   - For a new dataset, add a function `(split, batch_size) -> DataLoader` to `DATASETS`.
3. Train it (`python spikingjet.py train -n MyNet`), or copy its checkpoint to
   `models/pretrained_models/MyNet.th`.
4. Check it: `python spikingjet.py sites -n MyNet` (its fault sites), `python -m pytest -q tests`
   (runs it on random data), `python verify_snn.py --network MyNet` (clean accuracy, and that
   faults reach the output).
5. Run any tool: `python spikingjet.py campaign -n MyNet`, `python spikingjet.py ber -n MyNet`, ...

**Nothing has to be declared for faults.** Parameters and the outputs of ReLU / Conv / Linear
modules are found by themselves, and so are snntorch neurons (`snn.Leaky`, `snn.Synaptic`, ...):
their membrane potential and spikes become `<module>.potential` / `<module>.spike`, timestep by
timestep. See [`models/hybrid.py`](models/hybrid.py).

**Custom neurons** (not snntorch modules, e.g. a hand-written LIF loop) expose their state with
one call after every neuron update at timestep `t`:

```python
self.FaultInjector = None                          # in __init__
...
if self.FaultInjector is not None:
    self.FaultInjector.injection(spk, mem, "lif1", t)    # spikes FIRST, then membrane, layer name, step
```

`tests/test_injection_hook.py` checks this argument order for every registered network whose
checkpoint is present. `models/snn_sota.py` shows complete examples.

## Tests

```bash
python -m pytest -q tests     # unit tests + every network on random data (part of selftest.py)
python selftest.py [--full]   # every tool, see Setup
```
