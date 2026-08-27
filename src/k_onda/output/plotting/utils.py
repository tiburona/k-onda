from collections.abc import Iterable     
from dataclasses import replace, fields


def candidate_matches_selector(selector, candidate):
    for key, selected in selector.items():
        if key not in candidate:
            return False
            
        actual = candidate[key]

        if isinstance(selected, str):
            if actual != selected:
                return False
        elif isinstance(selected, Iterable):
            if actual not in selected:
                return False
        else:
            if actual != selected:
                return False

    return True


class UnsetType:
    __slots__ = ()

    def __repr__(self):
        return "UNSET"


UNSET = UnsetType()

def merge_dataclasses(instance, patch_instance=None, **changes):
    if patch_instance is not None:
        changes = {
            **changes, 
            **{
                field.name: getattr(patch_instance, field.name) 
                for field in fields(patch_instance) 
                if getattr(patch_instance, field.name) is not UNSET
                }
            }
        
    return replace(
        instance,
        **{name: value for name, value in changes.items() if value is not UNSET}
    )
