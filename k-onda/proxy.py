class Proxy:
    def __init__(self, target_instance, target_class):
        self._target_instance = target_instance
        self._target_class = target_class

    def __getattribute__(self, name):
        try:
            # Try to get attribute from the Proxy child (LFPAnimal or LFPUnit)
            return super().__getattribute__(name)
        except AttributeError:
            # If not found, try to get class-level attributes from target_class (Animal or Unit)
            return getattr(self._target_class, name)

    def __getattr__(self, name):
        # If not found in any of the above, get instance-level attributes from target_instance
        return getattr(self._target_instance, name)

    def __iter__(self):
        return iter(self._target_instance)