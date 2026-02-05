from .calculator import Calculator


class Shift(Calculator):

    def __init__(self, shift):
        self.config = {"shift": shift} 
    
    def _apply(self, data, shift):
        return data + shift 
    

class Scale(Calculator):

    def __init__(self, scaling_factor):
        self.config = {"scaling_factor": scaling_factor} 
    
    def _apply(self, data, scaling_factor):
        return data * scaling_factor
    

