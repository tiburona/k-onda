from .calculator import Calculator


class Shift(Calculator):

    def __init__(self, shift):
        self.shift = shift

    def _get_apply_kwargs(self, _):
        return {'shift': self.shift}
    
    def _apply(self, data, shift):
        return data + shift
    

class Scale(Calculator):

    def __init__(self, factor):
        self.factor = factor

    def _get_apply_kwargs(self, _):
        return {'factor': self.factor}
    
    def _apply(self, data, factor):
        return data * factor
    

