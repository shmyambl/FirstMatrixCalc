
import pytest
import numpy as np
from matrix_scratch import Matrix as ScratchMatrix
from matrix_numpy import Matrix as NumpyMatrix

# эта штука превращает данные в понятный питону список
def to_list(data):
    if hasattr(data, 'tolist'): return data.tolist()
    return [list(row) for row in data]

# запускаем тесты по очереди для обоих вариантов кода
@pytest.mark.parametrize("MatrixClass", [ScratchMatrix, NumpyMatrix])
class TestMatrixOperations:

    def test_transpose(self, MatrixClass):
        # проверяем как работает поворот матрицы
        matrix = MatrixClass([[1, 2], [3, 4]])
        assert to_list(matrix.transpose().data) == [[1, 3], [2, 4]]

    def test_divide_singular(self, MatrixClass):
        # проверяем что программа ругается если делить на плохую матрицу
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2], [2, 4]]) # тут определитель ноль
        with pytest.raises(ValueError): # ждем именно эту ошибку
            m1.divide(m2)

    def test_init_uneven_rows(self, MatrixClass):
        # проверяем что нельзя создать кривую матрицу с разной длиной строк
        with pytest.raises(ValueError):
            MatrixClass([[1, 2], [3]])

    def test_multiply_compatible(self, MatrixClass):
    a = MatrixClass([[1, 2, 3], [1, 2, 3]])
    b = MatrixClass([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert to_list(a.multiply(b).data) == [[30, 36, 42], [30, 36, 42]]

    def test_multiply_incompatible(self, MatrixClass):
    a = MatrixClass([[1, 2], [3, 4], [5, 6]])
    b = MatrixClass([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        a.multiply(b)

    def test_divide_compatible(self, MatrixClass):
    a = MatrixClass([[2, 5], [3, 4]])
    b = MatrixClass([[1, 2], [3, 4]])
    result = to_list(a.divide(b).data)
    expected = [[3.5, -0.5], [0.0, 1.0]]
    for r_row, e_row in zip(result, expected):
        for rv, ev in zip(r_row, e_row):
            assert abs(rv - ev) < 1e-3

    def test_divide_non_square(self, MatrixClass):
    a = MatrixClass([[1, 2], [3, 4]])
    b = MatrixClass([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        a.divide(b)
