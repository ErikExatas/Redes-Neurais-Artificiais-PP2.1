import numpy as np
from numpy.typing import DTypeLike


class Perceptron:

    _dtype: DTypeLike
    _weights: np.ndarray

    def __init__(self, input_size: int, weight_range: tuple[float, float] = (-0.5, 0.5), dtype: DTypeLike = np.float64):
        if (
            not isinstance(weight_range, tuple) or
            len(weight_range) != 2 or
            not all(isinstance(x, (int, float)) for x in weight_range) or
            weight_range[0] > weight_range[1]
        ):
            raise ValueError("weight_range deve ser uma tupla com dois valores numéricos (int ou float) onde o primeiro valor deve ser menor que o segundo.")


        self._dtype = dtype
        self._weights = np.random.uniform(-0.5, 0.5, input_size + 1).astype(self._dtype)
