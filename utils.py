import argparse

from networks import NETWORKS, load_network  # noqa: F401  (load_network re-exported)


def parse_args(argv=None):
    """Arguments of main.py (single-fault campaign). Options left out keep the value of --config,
    or the default of spikingjet/config.py."""
    parser = argparse.ArgumentParser(description='Run a single-fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network-name', '-n', type=str, required=True, choices=list(NETWORKS),
                        help='Target network (registered in networks.py)')
    parser.add_argument('--fault-mode', type=str, default=None, choices=['stuck-at', 'bit-flip'],
                        help='stuck-at = permanent hardware fault; bit-flip = single-event upset (default stuck-at)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed of the statistical fault sample (DATE09, default 51196). Different seeds give '
                             'independent samples.')
    parser.add_argument('--batch-size', '-b', type=int, default=None, help='Test set batch size (default 64)')
    parser.add_argument('--max-faults', type=int, default=None,
                        help='Cap the number of faults (quick check; the DATE09 statistics no longer hold)')
    parser.add_argument('--targets', nargs='+', default=None, choices=['params', 'neurons', 'activations'],
                        help='What gets faults (default: params neurons)')
    parser.add_argument('--neuron-persistence', default=None, choices=['permanent', 'transient'],
                        help='Neuron-state faults at every timestep (permanent, default) or at one random timestep')
    parser.add_argument('--config', '-c', default=None, help='experiment file (TOML), see experiments/')
    parser.add_argument('--set', '-s', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config key, e.g. campaign.bits=mantissa')
    parser.add_argument('--use-cuda', action='store_true', help='Use the GPU')
    parser.add_argument('--forbid-cuda', action='store_true', help='Disable CUDA (overrides --use-cuda)')
    from numerics import add_argument as add_tf32_argument
    add_tf32_argument(parser)
    return parser.parse_args(argv)
