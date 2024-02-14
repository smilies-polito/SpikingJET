import torch.nn as nn
from torch.nn import functional as F
import torch
import struct

def float_to_bin(num):
    # Pack the float into bytes
    packed = struct.pack('!f', num)
    # Convert bytes to binary string
    binary = ''.join(format(byte, '08b') for byte in packed)
    return binary

def int_to_bin(num):
    # Pack the float into bytes
    packed = struct.pack('!I', num)
    # Convert bytes to binary string
    binary = ''.join(format(byte, '08b') for byte in packed)
    return binary


def set_bits_to_one(indices, dim):

    faults_mask = [0]*dim
    

    for tensor_index, bit_index in indices:
        i = tensor_index[1]+3*tensor_index[0]
        faults_mask[i] += 2**bit_index
    
    #for i in range(dim):
    #    faults_mask[i] = struct.pack('!I', faults_mask[i])
    mask = torch.Tensor(faults_mask)
    mask = mask.view(torch.int)
    print(mask)

    return faults_mask

def flip_bits(potential_tensor, mask):
    for i in range(3):
        for y in range(3):
            float_list = []
            a = struct.pack('!f', potential_tensor[i,y])
            mask[i*3+y]
            for ba, bb in zip(a, mask[i*3+y]):
                float_list.append(ba ^ bb)

            potential_tensor[i,y] = struct.unpack('!f', bytes(float_list))[0]
    return potential_tensor    


def print_bin(tensor):
    # Iterate through each element in the tensor
    for num in tensor.flatten():
        bin_rep = float_to_bin(num)
        print(bin_rep)

potential_tensor = torch.tensor( [
    
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0]
            ]    
           ,
    dtype=torch.float32
)
print(potential_tensor)
pot_int = potential_tensor.view(torch.int)
print(pot_int)
mask= torch.tensor( [
    
                [0, 0, 2],
                [0, 537001984, 0],
                [16, 0, 0]
            ]    
           ,
    dtype=torch.int32
)
print(mask)
res = torch.bitwise_xor(pot_int, mask)
pot_float = res.view(torch.float)
print(pot_float)

'''
# Example usage:
indices = [((1,1), 20), ((1,0), 1), ((1,2), 8), ((1,2), 12)]
tensor_dimension = torch.Size([3,3])  # Example dimension



faults_mask = set_bits_to_one(indices, 9)
print("Faults mask:")
for i in range(9):
        print(faults_mask[i])

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
flipped_tensor = flip_bits(potential_tensor, faults_mask)
#result = torch.bitwise_xor(potential_tensor, mask_tensor)  #unfortunatly doesn't work with float tensor

print("input flip tens:")
print_bin(flipped_tensor)
print("Flipped Potential Tensor:")
print(flipped_tensor)
'''
