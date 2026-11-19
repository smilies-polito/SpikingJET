import torch

from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from networks import load_network
from utils import get_device, parse_args


def main(args):

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)
    # Exact FP32 unless --tf32 (numerics.py)
    from numerics import set_tf32
    print(f'Numerics: {set_tf32(args.tf32, deterministic=False)}')

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    print(f'Using device {device}')

    # Load the network
    network = load_network(args.network_name, device)

    # Load the dataset registered for the network (networks.py)
    from networks import get_loader
    loader = get_loader(args.network_name, 'test', args.batch_size)

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
                                                     seed=args.seed,
                                                     FaultInjectionManager = fault_injection_executor)

    # Run the faulty campaign (one fault at a time) with the requested fault mode
    fault_injection_executor.run_faulty_campaign(fault_list=fault_list,
                                                    fault_mode=args.fault_mode)
    

if __name__ == '__main__':
    main(args=parse_args())
