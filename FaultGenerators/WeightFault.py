class WeightFault:

    def __init__(self,
                 layer_name: str,
                 tensor_index: tuple,
                 bit: int,
                 value: int = None):
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit
        self.value = value
