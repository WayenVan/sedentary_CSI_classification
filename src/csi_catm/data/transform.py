import numpy as np

class DownSample(object):
    
    def __init__(self, t: int, h: int, w: int) -> None:
        self._t = t
        self._h = h
        self._w = w
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        #data: [t, ..., h, w]
        h_size = data.shape[-2]
        w_size = data.shape[-1]
        t_size = data.shape[0]
        
        assert h_size % self._h == 0
        assert w_size % self._w == 0
        assert t_size % self._t == 0

        return data[::self._t, ..., ::self._h, ::self._w]

class Norm:
    def __init__(self, ep=1e-8) -> None:
        self.ep = ep
    
    def __call__(self, data):
        #normalize
        std = np.std(data, (-1, -2), keepdims=True)
        mean = np.mean(data, (-1, -2), keepdims=True)
        data = (data - std) / (mean + self.ep)
        return data

class ApplyByKey():

    def __init__(self, key, transforms: list) -> None:
        self.key = key
        self.transforms = transforms

    def __call__(self, data):
        d = data[self.key]
        for t in self.transforms:
            d = t(d)
        data[self.key] = d
        return data
