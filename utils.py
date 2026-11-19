import argparse
import os

import torch

from networks import NETWORKS, load_network  # noqa: F401  (load_network re-exported)


def parse_args():
    """Arguments of main.py (single-fault campaign)."""
    parser = argparse.ArgumentParser(description='Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network-name', '-n', type=str, required=True, choices=list(NETWORKS),
                        help='Target network (registered in networks.py)')
    parser.add_argument('--fault-mode', type=str, default='stuck-at', choices=['stuck-at', 'bit-flip'],
                        help='stuck-at = permanent hardware fault; bit-flip = transient single-event upset')
    parser.add_argument('--seed', type=int, default=51196,
                        help='Seed of the statistical fault sample (DATE09). Different seeds give '
                             'independent samples.')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Test set batch size')
    parser.add_argument('--use-cuda', action='store_true', help='Use the GPU if available')
    parser.add_argument('--forbid-cuda', action='store_true', help='Disable CUDA (overrides --use-cuda)')
    from numerics import add_argument as add_tf32_argument
    add_tf32_argument(parser)
    return parser.parse_args()


def get_device(forbid_cuda: bool, use_cuda: bool) -> torch.device:
    if forbid_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if use_cuda:
            print('WARNING: cuda forcibly disabled even if --use-cuda is set')
        return torch.device('cpu')
    if use_cuda:
        if not torch.cuda.is_available():
            print('ERROR: cuda not available even if --use-cuda is set')
            exit(-1)
        return torch.device('cuda')
    return torch.device('cpu')
