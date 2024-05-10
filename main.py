import torch

from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets, load_DVSGesture_test_dataset, load_NMNIST_test_dataset, load_SHD_test_dataset

from utils import load_network, get_device, parse_args, UnknownNetworkException, plot_distribution


def main(args):

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    print(f'Using device {device}')

    # Load the network
    network = load_network(network_name=args.network_name,
                           device=device)
    
    network.eval()

    #This function plots the weights' distribution of the CNN
    plot_distribution(model=network, network=args.network_name, dataset="dataset_name") # Plots CNN, BatchNorm2d and Linear layers weight distribution for visualization purposes
    
    # Load the dataset
    if 'ResNet' in args.network_name:
        _, _, loader = load_CIFAR10_datasets(test_batch_size=args.batch_size)
    elif 'CSNN' in args.network_name:
        loader = load_DVSGesture_test_dataset(batch_size=args.batch_size)
    elif 'NMNIST' in args.network_name:
        loader = load_NMNIST_test_dataset(batch_size=args.batch_size)
    elif 'SHD' in args.network_name:
        loader = load_SHD_test_dataset(batch_size=args.batch_size)
    else:
        loader = load_ImageNet_validation_set(batch_size=args.batch_size,
                                              image_per_class=1)

    
    
    # Execute the fault injection campaign with the smart network
    fault_injection_executor = FaultInjectionManager(network=network,
                                                     network_name=args.network_name,
                                                     device=device,
                                                     loader=loader)

    network.FaultInjector = fault_injection_executor
    
    fault_manager = FaultListGenerator(network=network,
                                       network_name=args.network_name,
                                       device=device)

    #This function runs clean inferences on the golden dataset
    fault_injection_executor.run_clean(fault_manager)

    
    # Generate fault list
    fault_list = fault_manager.get_fault_list(load_fault_list=False,
                                                     save_fault_list=True,
                                                     FaultInjectionManager = fault_injection_executor)

    #This function runs fault injections
    fault_injection_executor.run_faulty_campaign(fault_list=fault_list)
    

if __name__ == '__main__':
    main(args=parse_args())
