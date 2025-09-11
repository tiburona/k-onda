from scipy.signal import hilbert

def apply_hilbert_to_padded_data(data, pad_len):
    result = hilbert(data)
    start = int(pad_len[0])
    stop  = -int(pad_len[1]) if pad_len[1] else None
    return result[..., start:stop]
