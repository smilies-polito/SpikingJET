import os
import argparse

import torch

from models.utils import load_from_dict
from models.CSNN import CSNN
from models.NMNIST import NMNIST
from models.SHD import SHD
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from scipy.stats import norm
from matplotlib.offsetbox import AnchoredText

class UnknownNetworkException(Exception):
    pass


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(description='Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--forbid-cuda', action='store_true',
                        help='Completely disable the usage of CUDA. This command overrides any other gpu options.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use the gpu if available.')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')
    parser.add_argument('--network-name', '-n', type=str,
                        required=True,
                        help='Target network',
                        choices=['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                 'DenseNet121', 'CSNN', 'NMNIST', 'SHD'])

    parsed_args = parser.parse_args()

    return parsed_args


def load_network(network_name: str,
                 device: torch.device) -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :return: The loaded network
    """

    if 'ResNet' in network_name:
        if network_name == 'ResNet20':
            network_function = resnet20
        elif network_name == 'ResNet32':
            network_function = resnet32
        elif network_name == 'ResNet44':
            network_function = resnet44
        elif network_name == 'ResNet56':
            network_function = resnet56
        elif network_name == 'ResNet110':
            network_function = resnet110
        elif network_name == 'ResNet1202':
            network_function = resnet1202
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of ResNet: {network_name}')

        # Instantiate the network
        network = network_function()

        # Load the weights
        network_path = f'models/pretrained_models/{network_name}.th'

        load_from_dict(network=network,
                       device=device,
                       path=network_path)
    elif 'DenseNet' in network_name:
        if network_name == 'DenseNet121':
            network = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of DenseNet: {network_name}')

    elif network_name == 'EfficientNet':
        network = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    elif 'CSNN' in network_name:
        # Create instance of network
        network = CSNN()

        # Load the weights
        network_path =  f'models/pretrained_models/{network_name}.th'
        
        load_from_dict(network=network,
                       device=device,
                       path=network_path)
    elif 'NMNIST' in network_name:
        network = NMNIST()
        
        # Load the weights
        network_path =  f'models/pretrained_models/{network_name}.th'

        load_from_dict(network=network,
                device=device,
                path=network_path)
    elif 'SHD' in network_name:
        network = SHD(device)

        #Load the weights
        network_path = f'models/pretrained_models/{network_name}.th'

        load_from_dict(network=network,
        device=device,
        path=network_path)

    else:
        raise UnknownNetworkException(f'ERROR: unknown network: {network_name}')

    # Send network to device and set for inference
    network.to(device)
    network.eval()

    return network


def get_device(forbid_cuda: bool,
               use_cuda: bool) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :return: The device where to perform the fault injection
    """

    # Disable gpu if set
    if forbid_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = 'cpu'
        if use_cuda:
            print('WARNING: cuda forcibly disabled even if set_cuda is set')
    # Otherwise, use the appropriate device
    else:
        if use_cuda:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = ''
                print('ERROR: cuda not available even if use-cuda is set')
                exit(-1)
        else:
            device = 'cpu'

    return torch.device(device)



#This function requires:
# -model: pytorch model
# -network: string of the model
# -dataset: name of the dataset

def plot_distribution(model, network, dataset):
    weights = [module.weight.flatten() for _, module in model.named_modules() if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.Linear)]
    concatenated_weights = torch.cat(weights).tolist()
    max_val=round(max(concatenated_weights),3)
    min_val=round(min(concatenated_weights),3)
    std_dev=round(np.std(concatenated_weights),3)
    mean=round(np.mean(concatenated_weights),3)
    # set up figure and axes
    f, ax = plt.subplots(1,1)
    #print(np.count_nonzero(concatenated_weights))
    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(-5, 5, 0.001)
    # Mean = 0, SD = 2.
    plt.hist(concatenated_weights, bins=1000, density=True)
    title_name="%s - %s" %(network, dataset)
    plt.title(title_name)
    plt.xlabel("x")
    plt.ylabel("PDF(x)")
    foldername="distributions_plots"
    pathname="./%s/%s-%s-weights-distrib.png" %(foldername, network, dataset)
    anchored_text = AnchoredText("Min=%s, Max=%s, Std Deviation=%s, Mean=%s" %(min_val, max_val, std_dev, mean), loc="upper right")
    ax.add_artist(anchored_text)
    plt.savefig(pathname)
