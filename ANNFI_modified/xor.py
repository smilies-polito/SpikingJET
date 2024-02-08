import torch.nn as nn
from torch.nn import functional as F
import torch

def create_zero_tensor(dimensions):
        return torch.zeros(dimensions, dtype=torch.float32)

def set_bits_to_one(original_tensor, indices):
   
    modified_tensor = original_tensor.clone()  # Clone the original tensor

    for tensor_index, bit_index in indices:
        # Convert the float32 number to its binary representation
        binary_representation = '{:032b}'.format(
            torch.tensor(original_tensor[tensor_index]).view(-1).type(torch.int32).item()
        )

        # Modify the bit at the specified index
        modified_representation = binary_representation[:-(bit_index + 1)] + '1' + binary_representation[-bit_index:]

        # Convert the modified binary representation back to float32
        modified_tensor[tensor_index] = torch.tensor(int(modified_representation, 2), dtype=torch.float32)

    return modified_tensor       

# Example usage:
indices = [((1,1,1,1), 20), ((1,1,1,2), 1), ((1,0,1,1), 8)]
tensor_dimension = torch.Size([3,2,3,3])  # Example dimension
zero_tensor = create_zero_tensor(tensor_dimension)
print(zero_tensor)
modified_tensor = set_bits_to_one(zero_tensor, indices)
print("Modified Tensor:")
print(modified_tensor)