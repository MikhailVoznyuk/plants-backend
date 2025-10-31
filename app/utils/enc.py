import numpy as np

def rle_encode(mask: np.ndarray) -> str:
    # mask: HxW boolean
    pixels = mask.flatten(order="F")
    pads = np.array([0, *pixels, 0], dtype=np.uint8)
    runs = np.where(pads[1:] != pads[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(map(str, runs.tolist()))
