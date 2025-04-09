import numpy as np


class Perceptron:

    _input_size: int
    _weights: np.ndarray

    def __init__(self, input_size: int, weight_range: tuple[float, float] = (-0.5, 0.5)):
        if (
            not isinstance(weight_range, tuple) or
            len(weight_range) != 2 or
            not all(isinstance(x, (int, float)) for x in weight_range) or
            weight_range[0] > weight_range[1]
        ):
            raise ValueError("`weight_range` deve ser uma tupla com dois valores numéricos (int ou float) onde o primeiro valor deve ser menor que o segundo.")

        self._input_size = input_size
        self._weights = np.random.uniform(*weight_range, input_size + 1).astype(np.float64)

    def predict(self, input_data: np.ndarray) -> int:
        """Realiza uma predição para uma única amostra.

        Parâmetros:
        input_data (np.ndarray): Vetor de entrada (sem o bias).

        Retorna:
        int: 0 ou 1, saída da função de ativação.
        """
        assert isinstance(input_data, np.ndarray), "`input_data` deve ser um `np.ndarray`."
        assert input_data.ndim == 1, "`input_data` deve ser um vetor 1D."
        assert input_data.shape[0] == self._input_size, f"Esperado vetor de {self._input_size} features, mas recebeu {input_data.shape[0]}."

        return self.__f(np.dot(self._weights, input_data))

    def train(self, X: np.ndarray, labels: np.ndarray ,learning_rate: float, max_epochs: int = 1000) -> int:
        """Treina o perceptron.

        Parâmetros:
        X (np.ndarray): Matriz de entradas (n, 2), sem bias.
        labels (np.ndarray): Saídas desejadas (0 ou 1), shape (n, 1).
        learning_rate (float): Taxa de aprendizado (positivo).
        max_epochs (int): Número máximo de épocas (positivo).

        Retorna:
        int: Época em que convergiu, ou -1 se não convergir.
        """
        assert isinstance(X, np.ndarray) and X.ndim == 2, "`X` deve ser uma matriz 2D do tipo `np.ndarray`."
        assert X.shape[1] == self._input_size, f"Esperado {self._input_size} features por amostra, mas recebeu {X.shape[1]}."

        assert isinstance(labels, np.ndarray), "`labels` deve ser um np.ndarray."
        assert labels.ndim == 2 and labels.shape[1] == 1, "`labels` deve ter forma (n, 1)."
        assert X.shape[0] == labels.shape[0], "`X` e `labels` devem ter o mesmo número de amostras."

        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, "`learning_rate` deve ser um número positivo."
        assert isinstance(max_epochs, int) and max_epochs > 0, "`max_epochs` deve ser um inteiro positivo."

        for i in range(max_epochs):
            predictions = self._batch_predict(X)
            errors = labels.ravel() - predictions

            if np.count_nonzero(errors) == 0:
                return i + 1

            # w <- w + η . X^T . (y_truth - y_pred)
            self._weights += learning_rate * (self._add_bias(X).T @ errors)

        return -1

    def _batch_predict(self, X: np.ndarray) -> np.ndarray:
        return (self._add_bias(X) @ self._weights >= 0).astype(int)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        return np.hstack((np.ones((X.shape[0], 1)), X))

    @staticmethod
    def __f(activation: float) -> int:
        return 1 if activation >= 0 else 0