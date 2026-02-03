from .calculator import Calculator


class Shift(Calculator):

    def __init__(self, shift):
        self.config = {"shift": shift} 
    
    def _apply(self, parent, shift):

        data = parent.data

        return data + shift 
    

class Scale(Calculator):

    def __init__(self, scaling_factor):
        self.config = {"scaling_factor": scaling_factor} 
    
    def _apply(self, parent, scaling_factor):

        data = parent.data
        # TODO: this probably needs to be sensitive to whether the input
        # is numpy, xarray, unit-aware xarray, etc.  I'm just trying to implement
        # this simplest possible thing at first.  
        return data * scaling_factor
    

