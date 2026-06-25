
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
