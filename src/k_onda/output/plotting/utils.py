from collections.abc import Iterable


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