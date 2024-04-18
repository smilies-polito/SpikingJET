import itertools
import os
import csv
from tqdm import tqdm
import numpy as np
from ast import literal_eval as make_tuple

from typing import Type

from FaultGenerators.WeightFault import WeightFault

from torch.nn import Module, Conv2d, Linear
import torch

import snntorch as snn



from FaultGenerators.utils import float32_bit_flip


class FaultListGenerator:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 module_classes: Type[Module] = (Conv2d, snn.Leaky, Linear, snn.Synaptic)): # 

        self.network = network
        self.network_name = network_name

        self.device = device

        # Name of the injectable layers
        injectable_layer_names = [name.split('.')[0] for name, module in self.network.named_modules()
                                  if isinstance(module, module_classes)]

        # List of the shape of all the layers that contain weight
        self.net_layer_shape = {name: param.shape for name, param in self.network.named_parameters()
                                if name.split('.')[0] in injectable_layer_names}
        #print(self.net_layer_shape)

        # List of the injectable params
        self.net_layer_params = {name: param for name, param in self.network.named_parameters()
                                 if name.split('.')[0] in injectable_layer_names}
        #print(self.net_layer_params)
            
        #print(self.network.state_dict())    
    @staticmethod
    def __compute_date_n(N: int,
                         p: float = 0.5,
                         e: float = 0.01,
                         t: float = 2.58):
        """
        Compute the number of faults to inject according to the DATE09 formula
        :param N: The total number of parameters
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: the number of fault to inject
        """
        return N / (1 + e ** 2 * (N - 1) / (t ** 2 * p * (1 - p)))

    def get_fault_list(self,
                                load_fault_list=False,
                                save_fault_list=True,
                                seed=51196,
                                p=0.5,
                                e=0.01,
                                t=2.58,
                                FaultInjectionManager=None):
        """
        Generate a fault list for the potential and beta of leaky layers and  according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: The fault list
        """

        for layer, size in FaultInjectionManager.size.items():

            size_no_batch = size[1:] #removing the batch correlated dimension
            self.net_layer_shape[layer+".potential"] = size_no_batch

        for layer, size in FaultInjectionManager.spike.items():

            size_no_batch = size[1:] #removing the batch correlated dimension
            self.net_layer_shape[layer+".spike"] = size_no_batch    

        # print(self.net_layer_shape)

        cwd = os.getcwd()
        fault_list_filename = f'{cwd}/output/fault_list/{self.network_name}'

        try:
            if load_fault_list:
                with open(f'{fault_list_filename}/{seed}_fault_list.csv', newline='') as f_list:
                    reader = csv.reader(f_list)

                    fault_list = list(reader)[1:]

                    fault_list = [WeightFault(layer_name=fault[1],
                                              tensor_index=make_tuple(fault[2]),
                                              bit=int(fault[-1])) for fault in fault_list]

                print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception and force the generation
            else:
                raise FileNotFoundError

        except FileNotFoundError:

            exhaustive_fault_list = []
            pbar = tqdm(self.net_layer_shape.items(), desc='Generating fault list', colour='green')
            for layer_name, layer_shape in pbar:
                # Add all the possible faults to the fault list
                k = np.arange(layer_shape[0])
                dim1 = np.arange(layer_shape[1]) if len(layer_shape) > 1 else [None]
                dim2 = np.arange(layer_shape[2]) if len(layer_shape) > 2 else [None]
                dim3 = np.arange(layer_shape[3]) if len(layer_shape) > 3 else [None]
                bits = np.arange(0, 32)

                exhaustive_fault_list = exhaustive_fault_list + list(
                    itertools.product(*[[layer_name], k, dim1, dim2, dim3, bits]))
                print(layer_name)
                print(len(list(itertools.product(*[[layer_name], k, dim1, dim2, dim3, bits]))))
            random_generator = np.random.default_rng(seed=seed)
            n = self.__compute_date_n(N=len(exhaustive_fault_list),
                                      p=p,
                                      e=e,
                                      t=t)
            
            fault_list = random_generator.choice(exhaustive_fault_list, int(n), replace=False)
            del exhaustive_fault_list
            fault_list = [WeightFault(layer_name=fault[0],
                                      tensor_index=tuple([int(i) for i in fault[1:-1]if i is not None]),
                                      bit=int(fault[-1])) for fault in fault_list]
            
            if save_fault_list:
                os.makedirs(fault_list_filename, exist_ok=True)
                with open(f'{fault_list_filename}/{seed}_fault_list.csv', 'w', newline='') as f_list:
                    writer_fault = csv.writer(f_list)

                    writer_fault.writerow(['Injection',
                                           'Layer',
                                           'TensorIndex',
                                           'Bit'])

                    golden_value_list = list()
                    faulty_value_list = list()
                    for index, fault in enumerate(fault_list):
                        
                        layer = fault.layer_name.split('.')[0]
                        location = fault.layer_name.split('.')[1]
                        if location != 'potential' and location != 'spike': #this controll checks if the fault is on potentian or spike, in this case jump this code cause fault value is not available
                            
                            # Get the golden value
                            golden_value = float(self.net_layer_params[fault.layer_name][fault.tensor_index])
                            # Get the faulty value
                            faulty_value = float32_bit_flip(golden_value=golden_value, bit=fault.bit)
                            # Append results to list
                            golden_value_list.append(golden_value)
                            faulty_value_list.append(faulty_value)
                        
                        else: 
                            golden_value_list.append(0)
                            faulty_value_list.append(0)
                            if layer not in FaultInjectionManager.faults.keys(): 
                                FaultInjectionManager.faults[layer] = []
                            FaultInjectionManager.faults[layer].append([fault.tensor_index, fault.bit])
                        # Write fault list to csv
                        writer_fault.writerow([index, fault.layer_name, fault.tensor_index, fault.bit])    

                    # Write fault information to numpy
                    np.savez(f'{fault_list_filename}/{seed}_weights', golden=golden_value_list, faulty=faulty_value_list)
            
            #self.network.compute_mask()
            print('Fault List Generated')

        return fault_list


  