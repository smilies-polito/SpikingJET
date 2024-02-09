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

def flip_bits(modified_tensor, potential_tensor):
    """
    Flips the bits in the potential tensor corresponding to the set bits in the modified tensor.
    
    Parameters:
        modified_tensor (torch.Tensor): The modified tensor with bits set to 1.
        potential_tensor (torch.Tensor): The tensor to be modified based on the modified tensor.
    
    Returns:
        torch.Tensor: The modified potential tensor with flipped bits.
    """
    potential_int = potential_tensor.type(torch.int32)
    
    # Create a tensor with all bits set to 1
    ones_tensor = torch.ones_like(modified_tensor, dtype=torch.int32)
    
    # Perform a bitwise XOR operation between the ones tensor and the modified tensor
    # This will flip the bits in the potential tensor where the modified tensor has 1s
    flipped_bits_int = potential_int ^ (ones_tensor << modified_tensor.type(torch.int32))
    
    # Convert the modified potential tensor back to float32
    flipped_bits_tensor = flipped_bits_int.type(torch.float32)
    
    return flipped_bits_tensor

def print_bin(tensor):
    # Iterate through each element in the tensor
    for num in tensor.flatten():
        # Convert the element to its binary representation
        binary_repr = bin(torch.tensor(num).view(-1).long().item())
        # Print the binary representation, slicing off the first two characters (which are '0b')
        print(binary_repr[2:].zfill(tensor.element_size() * 8))


# Example usage:
indices = [((1,1), 20), ((1,0), 1), ((1,2), 8)]
tensor_dimension = torch.Size([3,3])  # Example dimension
zero_tensor = create_zero_tensor(tensor_dimension)
print(zero_tensor)
print_bin(zero_tensor)

modified_tensor = set_bits_to_one(zero_tensor, indices)
print("Modified Tensor:")
print(modified_tensor)
print_bin(modified_tensor)
potential_tensor = torch.tensor( [
    
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0]
            ]    
           ,
    dtype=torch.float32
)
print("input tens:")
print_bin(potential_tensor)
# Flip the bits in the potential tensor based on the modified tensor
#flipped_potential_tensor = flip_bits(modified_tensor, potential_tensor)
result = torch.bitwise_xor(potential_tensor, modified_tensor)

print("input flip tens:")
print_bin(result)
print("Flipped Potential Tensor:")
print(result)
