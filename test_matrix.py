import pytest
import numpy as np
from matrix_scratch import Matrix as ScratchMatrix
from matrix_numpy import Matrix as NumpyMatrix

# вспомогательная функция для перевода данных в обычные списки
def to_list(data):
    if hasattr(data, 'tolist'): return data.tolist()
    return [list(row) if isinstance(row, (list, tuple, np.ndarray)) else row for row in data]

@pytest.mark.parametrize("MatrixClass", [ScratchMatrix, NumpyMatrix])
class TestMatrixOperations:
    # проверка поворота матрицы
    def test_transpose(self, MatrixClass):
        matrix = MatrixClass([[1, 2], [3, 4]])
        assert to_list(matrix.transpose().data) == [[1, 3], [2, 4]]

    # проверка обычного умножения
    def test_multiply_compatible(self, MatrixClass):
        a = MatrixClass([[1, 2], [3, 4]])
        b = MatrixClass([[1, 0], [0, 1]])
        assert to_list(a.multiply(b).data) == [[1, 2], [3, 4]]

    # проверка правильного деления
    def test_divide_compatible(self, MatrixClass):
        m1 = MatrixClass([[2, 5], [3, 4]])
        m2 = MatrixClass([[1, 2], [3, 4]])
        result = m1.divide(m2)
        expected = [[3.5, -0.5], [0.0, 1.0]]
        res_list = to_list(result.data)
        for i in range(len(expected)):
            assert res_list[i] == pytest.approx(expected[i])

    # проверка защиты от неквадратных матриц
    def test_divide_non_square(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2, 3]])
        with pytest.raises(ValueError):
            m1.divide(m2)

    # проверка защиты от деления на нулевую (сингулярную) матрицу
    def test_divide_singular(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2], [2, 4]])
        with pytest.raises(ValueError): # требуем строго ошибку ValueError
            m1.divide(m2)

    # проверка защиты от кривых данных на входе
    def test_init_uneven_rows(self, MatrixClass):
        with pytest.raises(ValueError):
            MatrixClass([[1, 2], [3]])
