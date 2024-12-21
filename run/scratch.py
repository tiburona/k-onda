import pickle

def safe_load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        print(f"Failed to load object due to missing module: {e}")
        return None

data = safe_load_pickle('/Users/katie/likhtik/IG_INED_Safety_Recall/IG160/units_from_phy.pkl')
