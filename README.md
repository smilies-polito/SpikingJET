# SpikingJET

Fault injection for spiking (SNN) and conventional (CNN / DNN) neural networks in PyTorch.

| Tool | What it measures | Command |
|---|---|---|
| **Single-fault campaign** | one fault at a time (bit-flip or stuck-at) in weights and, for SNNs, in membrane potentials and spikes; statistical sample (DATE'09) | `main.py` → `analyse.py` |
| **BER sweep** | many random faults at once at a given bit error rate, accuracy vs BER | `ber_eval.py` → `ber_report.py` |
| **Hardware fault model** | faults in the memories of a neuromorphic chip (quantized weights, delays, neuron parameters, membrane state, spikes), single faults or BER sweeps | `deploy_shd.py` → `hw_campaign.py` |

Every tool works on any network registered in [`networks.py`](networks.py).

## Contents

```
networks.py              registry of networks and datasets  <- the only file to edit for a new network
main.py                  single-fault campaign
analyse.py, pt_to_csv.py campaign results -> CSV
ber_eval.py, ber_report.py   BER sweep and report
deploy_shd.py, hw_campaign.py, hwfault/   hardware fault model (quantized deployment form)
train_snn.py, train_shd_delays.py, train_cnn.py   training
verify_snn.py            check a network before a campaign
FaultInjectionManager.py, FaultGenerators/        injection engine
models/                  architectures; models/pretrained_models/ checkpoints
faulty_snn.def           Singularity / Apptainer container
tests/                   pytest
```

## Setup

Container (recommended):

```bash
sudo singularity build faulty_snn.sif faulty_snn.def
singularity shell --nv faulty_snn.sif        # then run the commands below from the repo root
```

Or with pip (Python >= 3.10): `pip install -r requirements.txt`.

Datasets are downloaded into `./data` on first use (SHD, N-MNIST and DVSGesture with `tonic`,
CIFAR-10 with `torchvision`). All results go to `./output` (single-fault campaigns) or to the
`--out` path you give.

## Networks included

| Name | Type | Dataset | Checkpoint |
|---|---|---|---|
| `SHD_DELAY` | SNN with learnable delays, 2 x 256 LIF | SHD | included |
| `NMNIST_CSNN` | convolutional SNN | N-MNIST | included |
| `NMNIST` | SNN (snntorch) | N-MNIST | included |
| `CSNN` | convolutional SNN (snntorch) | DVSGesture | included |
| `SHD_RLIF` | recurrent SNN | SHD | train with `train_snn.py` |
| `ResNet20/32/44/56`, `ResNet18`, `VGG9` | CNN | CIFAR-10 | train with `train_cnn.py` |

## 1. Single-fault campaign

```bash
python main.py -n SHD_DELAY --use-cuda --fault-mode bit-flip     # or --fault-mode stuck-at
python analyse.py -n SHD_DELAY -b 64 -fl 51196_fault_list.csv    # per-layer results
python pt_to_csv.py -n SHD_DELAY -b 64                           # optional: raw outputs as CSV
```

- The fault list is a statistical sample (DATE'09 formula: 3 % error margin, 99 % confidence) of
  all (parameter, bit) pairs of the weights and, for SNNs, of the neuron states. `--seed` gives an
  independent sample.
- Results:
  - `output/fault_list/<network>/<seed>_fault_list.csv`: the faults;
  - `.../<seed>_fault_list_result_batch_size_<b>.csv`: per layer, the fraction of fully masked
    faults, critical faults (`SDC_1`, prediction changed), output-score deviations
    (`SDC_0_5%` ... `SDC_20`), and the faulty / clean accuracy (`facc`, `acc`).
- `-b` must match the `--batch-size` of `main.py` (default 64).

## 2. BER sweep (many faults at once)

```bash
python ber_eval.py --network SHD_DELAY NMNIST_CSNN --bers 1e-7 1e-6 1e-5 1e-4 \
    --bits all mantissa --realizations 10 --out ber_results/snn.csv
python ber_report.py ber_results/snn.csv --plot ber_results/snn.png
```

- Every stored FP32 bit flips with probability BER; each realization runs the whole test set.
- `--bits`: `all`, `mantissa`, `exponent`, `sign` or `low-mantissa`.
- `--mode stuck-at` instead of bit flips.
- The fault map depends only on (seed, BER, bits, realization), so two networks with the same
  shapes see the same faults.
- `ber_report.py` prints accuracy vs BER and the tolerable BER at a 1 pp loss (`--loss`).

## 3. Hardware fault model (neuromorphic memories)

The network is first converted to its deployment form: integer codes in the formats of a precision
profile. Faults then hit the memories of the chip:

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
  accuracy loss, confidence loss and the number of faults per class.

**Single-fault catalog:**

```bash
python hw_campaign.py faults  --checkpoint models/pretrained_models/SHD_DELAY_deployed.th \
    --fraction 0.01 --out hw_results/faults.csv.gz
python hw_campaign.py catalog --checkpoint models/pretrained_models/SHD_DELAY_deployed.th \
    --faults hw_results/faults.csv.gz --out hw_results/catalog
```

- The catalog measures each fault separately. It is resumable: run it again to continue.
- Output: `hw_results/catalog/records.csv`, one row per fault.

**Options:**
- `--persistence`: `permanent`, `transient`, `mixed` (default), or JSON per class, e.g.
  `'{"M1": "permanent", "M3": "transient"}'`.
- Precision profiles live in `hwfault/profiles/*.toml`: `default` (int8 weights, 16-bit fixed-point
  membrane), `int4w`, `fp32`. Copy one to define your own and pass `--profile <file>`.

## Train a network

```bash
python train_shd_delays.py                     # SHD_DELAY (150 epochs, needs dcls)
python train_snn.py --network NMNIST_CSNN      # or SHD_RLIF
python train_cnn.py -n ResNet20 --epochs 200   # any CNN of networks.py (CIFAR-10)
python verify_snn.py --network SHD_DELAY       # clean accuracy + injection check before a campaign
```

Checkpoints are written to the path registered in `networks.py`, where every tool finds them.

## Add your own network (SNN, CNN or any PyTorch model)

1. Put the model in `models/my_net.py`. `forward(x)` must return one score vector per sample,
   shape `[batch, classes]`.
2. Register it in `networks.py`:

   ```python
   "MyNet": Net(_cls("my_net", "MyNet"), f"{PRE}/MyNet.th", "cifar10"),
   ```

   - The arguments are: builder, checkpoint, dataset.
   - The checkpoint is a `state_dict`, or `{"state_dict": ...}`.
   - For a new dataset, add a function `(split, batch_size) -> DataLoader` to `DATASETS`.
3. Train it, or copy its checkpoint to `models/pretrained_models/MyNet.th`.
4. Run any tool: `python main.py -n MyNet --use-cuda`, `python ber_eval.py --network MyNet ...`

**Which parameters get faults:** weight faults go into every top-level module that is, or
contains, a `Conv2d`, `Linear`, `snn.Leaky` or `snn.Synaptic`. Nothing has to be declared.

**Neuron faults (SNNs):** to also inject into membrane potentials and spikes:
- give the model an attribute `self.FaultInjector = None`;
- give it the number of timesteps in `self.num_steps`;
- after every neuron update at timestep `t`, call

  ```python
  if self.FaultInjector is not None:
      self.FaultInjector.injection(spk, mem, "lif1", t)    # spikes, membrane, layer name, step
  ```

The layer name labels the faults in the results. `models/snn_sota.py` shows complete examples.

## Tests

```bash
python -m pytest -q tests
```
