from .calculator import Calculator, with_key_access


class Shift(Calculator):

    def __init__(self, shift):
        self.shift = shift

    def _get_distinctive_apply_kwargs(self, _):
        return {'shift': self.shift}
    
    @with_key_access
    def _apply(self, data, shift):
        return data + shift
    

class Scale(Calculator):

    def __init__(self, factor):
        self.factor = factor

    def _get_distinctive_apply_kwargs(self, _):
        return {'factor': self.factor}
    
    @with_key_access
    def _apply(self, data, factor):
        return data * factor
    

