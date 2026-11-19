import struct


def float32_bit_flip(golden_value: float,
                       bit: int) -> float:
    """
    Inject a bit-flip on a data represented as float32
    :param golden_value: the value to bit-flip
    :param bit: the bit where to perform the bit-flip
    :return: The value of the bit-flip on the golden value
    """
    float_list = []
    a = struct.pack('!f', golden_value)
    b = struct.pack('!I', int(2. ** bit))
    for ba, bb in zip(a, b):
        float_list.append(ba ^ bb)

    faulted_value = struct.unpack('!f', bytes(float_list))[0]

    return faulted_value
