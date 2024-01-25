import struct


class WeightFaultInjector:

    def __init__(self, network):

        self.network = network

        self.layer_name = None
        self.tensor_index = None
        self.bit = None

        self.golden_value = None


    def __inject_fault(self, layer_name, tensor_index, bit, value=None):
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit

        self.golden_value = float(self.network.state_dict()[self.layer_name][self.tensor_index])

        # If the value is not set, then we are doing a bit-flip
        if value is None:
            faulty_value = self.__float32_bit_flip()
        else:
            faulty_value = self.__float32_stuck_at(value)

        self.faulty_value = faulty_value

        self.network.state_dict()[self.layer_name][self.tensor_index] = faulty_value


    def __float32_bit_flip(self):
        """
        Inject a bit-flip on a data represented as float32
        :return: The value of the bit-flip on the golden value
        """
        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            float_list.append(ba ^ bb)

        faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return faulted_value

    def __float32_stuck_at(self,
                           value: int):
        """
        Inject a stuck-at fault on a data represented as float32
        :param value: the value to use as stuck-at value
        :return: The value of the bit-flip on the golden value
        """
        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            if value == 1:
                float_list.append(ba | bb)
            else:
                float_list.append(ba & (255 - bb))

        faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return faulted_value

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        self.network.state_dict()[self.layer_name][self.tensor_index] = self.golden_value

    def inject_bit_flip(self,
                        layer_name: str,
                        tensor_index: tuple,
                        bit: int):
        """
        Inject a bit-flip in the specified layer at the tensor_index position for the specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        """
        self.__inject_fault(layer_name=layer_name,
                            tensor_index=tensor_index,
                            bit=bit)

    def inject_stuck_at(self,
                        layer_name: str,
                        tensor_index: tuple,
                        bit: int,
                        value: int):
        """
        Inject a stuck-at fault to the specified value in the specified layer at the tensor_index position for the
        specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        :param value: The stuck-at value to set
        """
        self.__inject_fault(layer_name=layer_name,
                            tensor_index=tensor_index,
                            bit=bit,
                            value=value)
