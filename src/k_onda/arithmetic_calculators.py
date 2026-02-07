from .calculator import Calculator


class Shift(Calculator):

    def __init__(self, shift):
        self.shift = shift
    
    def _apply(self, data):
        return data + self.shift
    

class Scale(Calculator):

    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor
    
    def _apply(self, data):
        return data * self.scaling_factor
    

