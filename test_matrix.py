import pytest
import numpy as np
from matrix_scratch import Matrix as ScratchMatrix
from matrix_numpy import Matrix as NumpyMatrix

# вспомогательная функция для перевода данных в обычные списки в среде numpy
def to_list(data):
    if hasattr(data, 'tolist'): return data.tolist()  # если это массив numpy, используем его метод
    return [list(row) if isinstance(row, (list, tuple, np.ndarray)) else row for row in data]

@pytest.mark.parametrize("MatrixClass", [ScratchMatrix, NumpyMatrix])
class TestMatrixOperations:
    # проверка транспонирования (поворота) матрицы
    def test_transpose(self, MatrixClass):
        matrix = MatrixClass([[1, 2], [3, 4]])
        assert to_list(matrix.transpose().data) == [[1, 3], [2, 4]]

    # проверка обычного умножения двух матриц
    def test_multiply_compatible(self, MatrixClass):
        a = MatrixClass([[1, 2], [3, 4]])
        b = MatrixClass([[1, 0], [0, 1]])
        assert to_list(a.multiply(b).data) == [[1, 2], [3, 4]]

    # проверка корректности деления через обратную матрицу
    def test_divide_compatible(self, MatrixClass):
        m1 = MatrixClass([[2, 5], [3, 4]])
        m2 = MatrixClass([[1, 2], [3, 4]])
        result = m1.divide(m2)
        expected = [[3.5, -0.5], [0.0, 1.0]]  # эталонный результат
        res_list = to_list(result.data)
        for i in range(len(expected)):
            assert res_list[i] == pytest.approx(expected[i])  # сравниваем с учетом мелких погрешностей

    # проверка защиты от деления неквадратных матриц
    def test_divide_non_square(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2, 3]])
        with pytest.raises(ValueError):  # ожидаем ошибку ValueError
            m1.divide(m2)

    # проверка защиты от деления на вырожденную (сингулярную) матрицу
    def test_divide_singular(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2], [2, 4]])
        with pytest.raises(ValueError):  # здесь обязательно должна выпасть ошибка
            m1.divide(m2)

    # проверка защиты от некорректных данных (разная длина строк)
    def test_init_uneven_rows(self, MatrixClass):
        with pytest.raises(ValueError):
            MatrixClass([[1, 2], [3]])
