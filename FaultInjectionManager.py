import os
import shutil
import time
import math
import random
from datetime import timedelta

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm

from FaultGenerators.WeightFaultInjector import WeightFaultInjector


from FaultGenerators.utils import float32_bit_flip


class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 loader: DataLoader):
        
        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.transient = False
        self.one = True
        
        self.inject = False
        self.current = None
        self.size = {}
        self.spike = {}
        self.faults = {}
        self.mask = {}
        self.num_step = 25


        # The clean output of the network after the first run
        self.clean_output_scores = list()
        self.clean_output_indices = list()

        # The weight fault injector
        self.weight_fault_injector = WeightFaultInjector(self.network)

        # The output dir
        self.label_output_dir = f'output/{self.network_name}/pt/label/batch_size_{self.loader.batch_size}'
        self.clean_output_dir = f'output/{self.network_name}/pt/clean/batch_size_{self.loader.batch_size}'
        self.faulty_output_dir = f'output/{self.network_name}/pt/faulty/batch_size_{self.loader.batch_size}'
        
        # Create the output dir
        os.makedirs(self.label_output_dir, exist_ok=True)
        os.makedirs(self.clean_output_dir, exist_ok=True)
        os.makedirs(self.faulty_output_dir, exist_ok=True)
        
    def run_clean(self, fault_manager):
        """
        Run a clean inference of the network
        :return: A string containing the formatted time elapsed from the beginning to the end of the fault injection
        campaign
        """
        self.inject = False
        with torch.no_grad():

            # Start measuring the time elapsed
            start_time = time.time()

            # Cycle all the batches in the data loader
            pbar = tqdm(self.loader,
                        colour='green',
                        desc=f'Clean Run',
                        ncols=shutil.get_terminal_size().columns)

            for batch_id, batch in enumerate(pbar):
                #print(batch_id)
                data, label = batch
                #print(len(label)) total of 10000 images
                data = data.to(self.device)

                # Run inference on the current batch
                scores, indices = self.__run_inference_on_batch(data=data)

                # Save the output
                torch.save(scores, f'{self.clean_output_dir}/batch_{batch_id}.pt')
                torch.save(label, f'{self.label_output_dir}/batch_{batch_id}.pt')

                # Append the results to a list
                self.clean_output_scores.append(scores)
                self.clean_output_indices.append(indices)

        # Stop measuring the time
        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed))
        

    def run_faulty_campaign(self,
                            fault_list: list,
                            different_scores: bool = False) -> str:
        """
        Run a faulty injection campaign for the network
        :param fault_list: list of fault to inject
        :param different_scores: Default False. If True, compare the faulty scores with the clean scores, otherwise
        compare the top-1 indices
        :return: A string containing the formatted time elapsed from the beginning to the end of the fault injection
        campaign
        """

        total_different_predictions = 0
        total_predictions = 0

        self.inject = True
        with torch.no_grad():

            # Start measuring the time elapsed
            start_time = time.time()

            pbar = tqdm(self.loader,
                        total=len(self.loader) * len(fault_list),
                        colour='green',
                        desc=f'Fault Injection campaign',
                        ncols=shutil.get_terminal_size().columns * 2)
            # Cycle all the batches in the data loader
            for batch_id, batch in enumerate(self.loader):
                data, _ = batch
                data = data.to(self.device)
                
                # Inject all the faults in a single batch
                for fault_id, fault in enumerate(fault_list):
                    
                    if fault.layer_name.split('.')[1] != 'potential' and fault.layer_name.split('.')[1] != 'spike':
                        # Inject faults in the weight
                        self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                        self.current = None
                    else:
                        self.current = [fault.layer_name.split('.')[0], fault.layer_name.split('.')[1], fault.tensor_index, fault.bit, random.randint(0,24)]
                    
                    if (fault.layer_name.split('.')[1] != 'potential' and fault.layer_name.split('.')[1] != 'spike') or self.one == True:
                        # Run inference on the current batch
                        faulty_scores, faulty_indices = self.__run_inference_on_batch(data=data)

                        # Save the output
                        torch.save(faulty_scores, f'{self.faulty_output_dir}/fault_{fault_id}_batch_{batch_id}.pt')
                        
                        # Measure the different predictions
                        if different_scores:
                            different_predictions = int(torch.ne(faulty_scores,
                                                                self.clean_output_scores[batch_id]).sum())
                        else:
                            different_predictions = int(torch.ne(torch.Tensor(faulty_indices),
                                                                torch.Tensor(self.clean_output_indices[batch_id])).sum())

                        # Measure the loss in accuracy
                        total_different_predictions += different_predictions
                        total_predictions += len(batch[0])
                        different_predictions_percentage = 100 * total_different_predictions / total_predictions
                        pbar.set_postfix({'Different': f'{different_predictions_percentage:.4f}%'})

                    if fault.layer_name.split('.')[1] != 'potential' and fault.layer_name.split('.')[1] != 'spike':
                        # Restore the golden value
                        self.weight_fault_injector.restore_golden()

                    # Update the progress bar
                    pbar.update(1)
        # Stop measuring the time
        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed))


    def __run_inference_on_batch(self,
                                 data: torch.Tensor):
        """
        Rim a fault injection on a single batch
        :param data: The input data from the batch
        :return: a tuple (scores, indices) where the scores are the vector score of each element in the batch and the
        indices are the argmax of the vector score
        """
        
        # Execute the network on the batch
        network_output = self.network(data)
        prediction = torch.topk(network_output, k=1)

        # Get the score and the indices of the predictions
        prediction_scores = network_output.cpu()
        prediction_indices = [int(fault) for fault in prediction.indices]

        return prediction_scores, prediction_indices

    def __inject_fault_on_weight(self,
                                 fault,
                                 fault_mode='stuck-at'):
        """
        Inject a fault in one of the weight of the network
        :param fault: The fault to inject
        :param fault_mode: Default 'stuck-at'. One of either 'stuck-at' or 'bit-flip'. Which kind of fault model to
        employ
        """

        if fault_mode == 'stuck-at':
            self.weight_fault_injector.inject_stuck_at(layer_name=f'{fault.layer_name}',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,
                                                       value=fault.value)

        elif fault_mode == 'bit-flip':
            self.weight_fault_injector.inject_bit_flip(layer_name=f'{fault.layer_name}',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,)
        else:
            print('FaultInjectionManager: Invalid fault mode')
            quit()

    def flatten_layer_dim(self, layer):
      '''
      For a given layer (Multi dimensional array) compute
      the dimension of the flatten array (Mono dimensional array) 
      '''
      n = 1
      for dim in self.size[layer]:
        n = n*dim  
      return n  

    def compute_flatten_index(self, layer, index_arr):
      '''
      Given an index of the layer (Multi dimensional array) return
      the corresponding index of the flatten array
      '''
      index = 0
      for n in range(len(index_arr)):
        i = n+1
        tmp =  index_arr[-i]
        for y in range(n):
          i2 = y+1
          tmp = tmp *self.size[layer][-i2]
        index = index + tmp 
      return index   
 
    def compute_mask(self):
      '''
      Compute the mask to perform the xor injection
      '''
      for layer, faults in self.faults.items():
        if layer not in self.mask.keys():
          self.mask[layer] = torch.zeros(self.size[layer], dtype=torch.int32)
        for tensor_index, bit_index in faults:
          
          self.mask[layer][tensor_index] += 2**bit_index
          

    def injection_dict(self, pot, spike, layer, pot_or_spike, tensor_index, bit_index):
        '''
        inject faults using the dictionary
        '''
        if pot_or_spike == "potential":
          for i in range(self.size[layer][0]):
            golden_value = float(pot[(i,) + tensor_index])
            faulty_value = float32_bit_flip(golden_value=golden_value, bit=bit_index)
            pot[(i,) + tensor_index] = faulty_value
        else: #to be modified
          for i in range(self.size[layer][0]):
            golden_value = float(pot[(i,) + tensor_index])
            faulty_value = float32_bit_flip(golden_value=golden_value, bit=bit_index)
            pot[(i,) + tensor_index] = faulty_value     

    def injection_xor(self, pot, layer):    
      '''
        inject faults using the XOR mask
      '''
      pot_int = pot.view(torch.int)
      pot_int_incjeted = torch.bitwise_xor(pot_int, self.mask[layer])
      pot = pot_int_incjeted.view(torch.float)


    def injection(self, spike, pot, layer, curr_step):
      if self.inject:
        if self.current != None:  
          if self.one == False:
            self.injection_xor(pot, layer)
          else:  
            if self.current[0] == layer:
              if self.transient == False or (self.transient and curr_step == self.current[4]):
                
                self.injection_dict(pot, spike, layer, self.current[1], self.current[2], self.current[3])
                
                
      else:
        self.size[layer] = pot.shape
        self.spike[layer] = spike.shape