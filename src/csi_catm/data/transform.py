import numpy as np

class DownSample(object):
    
    def __init__(self, h: int, w: int) -> None:
        self._h = h
        self._w = w
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        #data: [..., h, w]
        h_size = data.shape[-2]
        w_size = data.shape[-1]
        
        assert h_size % self._h == 0
        assert w_size % self._w == 0
        return data[..., ::self._h, ::self._w]