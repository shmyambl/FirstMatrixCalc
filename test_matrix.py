import pytest
import numpy as np

# Импортируем ваши классы
from matrix_scratch import Matrix as ScratchMatrix
from matrix_numpy import Matrix as NumpyMatrix

def to_list(data):
    if hasattr(data, 'tolist'):
        return data.tolist()
    return [list(row) if isinstance(row, (list, tuple, np.ndarray)) else row for row in data]

@pytest.mark.parametrize("MatrixClass", [ScratchMatrix, NumpyMatrix])
class TestMatrixOperations:

    def test_transpose(self, MatrixClass):
        data = [[1, 2, 3], [1, 2, 3]]
        expected = [[1, 1], [2, 2], [3, 3]]
        matrix = MatrixClass(data)
        result = matrix.transpose()
        assert to_list(result.data) == expected

    def test_multiply_compatible(self, MatrixClass):
        a = [[1, 2, 3], [1, 2, 3]]
        b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = [[30, 36, 42], [30, 36, 42]]
        matrix_a = MatrixClass(a)
        matrix_b = MatrixClass(b)
        result = matrix_a.multiply(matrix_b)
        assert to_list(result.data) == expected

    def test_multiply_incompatible(self, MatrixClass):
        a = [[1, 2], [3, 4], [5, 6]]
        b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrix_a = MatrixClass(a)
        matrix_b = MatrixClass(b)
        with pytest.raises(ValueError):
            matrix_a.multiply(matrix_b)

    def test_divide_compatible(self, MatrixClass):
        m1 = [[2, 5], [3, 4]]
        m2 = [[1, 2], [3, 4]]
        expected = [[3.5, -0.5], [0.0, 1.0]]
        matrix_1 = MatrixClass(m1)
        matrix_2 = MatrixClass(m2)
        result = matrix_1.divide(matrix_2)
        
        res_list = to_list(result.data)
        for i in range(len(expected)):
            assert res_list[i] == pytest.approx(expected[i])

    def test_divide_non_square(self, MatrixClass):
        m1 = [[1, 2], [3, 4]]
        m2 = [[1, 2, 3], [4, 5, 6]]
        matrix_1 = MatrixClass(m1)
        matrix_2 = MatrixClass(m2)
        with pytest.raises(ValueError):
            matrix_1.divide(matrix_2)

    def test_divide_singular(self, MatrixClass):
        m1 = [[1, 2], [3, 4]]
        m2 = [[1, 2], [2, 4]]
        matrix_1 = MatrixClass(m1)
        matrix_2 = MatrixClass(m2)
        if MatrixClass.__name__ == 'NumpyMatrix':
            with pytest.raises((ValueError, np.linalg.LinAlgError)):
                matrix_1.divide(matrix_2)
        else:
            with pytest.raises(ValueError):
                matrix_1.divide(matrix_2)

    def test_init_uneven_rows(self, MatrixClass):
        data = [[1, 2, 3], [4, 5]]
        with pytest.raises(ValueError):
            MatrixClass(data)
