
import pytest  # библиотека для удобного запуска тестов
import numpy as np  # библиотека для математических операций

# импортируем ваши классы матриц: один на чистом питоне, второй на numpy
from matrix_scratch import Matrix as ScratchMatrix
from matrix_numpy import Matrix as NumpyMatrix

# вспомогательная функция: превращает любые данные в обычный список списков для сравнения
def to_list(data):
    if hasattr(data, 'tolist'): return data.tolist() # если это объект numpy, используем его метод tolist()
    return [list(row) for row in data] # иначе просто перебираем строки

# этот декоратор заставляет pytest запустить все тесты ниже дважды (для каждой реализации)
@pytest.mark.parametrize("MatrixClass", [ScratchMatrix, NumpyMatrix])
class TestMatrixOperations:

    # тест на транспонирование (поворот матрицы)
    def test_transpose(self, MatrixClass):
        matrix = MatrixClass([[1, 2], [3, 4]]) # создаем матрицу
        result = matrix.transpose() # вызываем поворот
        assert to_list(result.data) == [[1, 3], [2, 4]] # проверяем, что строки стали столбцами

    # тест на обычное деление матриц
    def test_divide_compatible(self, MatrixClass):
        m1 = MatrixClass([[2, 5], [3, 4]]) # делимое
        m2 = MatrixClass([[1, 2], [3, 4]]) # делитель
        result = m1.divide(m2) # выполняем деление
        expected = [[3.5, -0.5], [0.0, 1.0]] # ожидаемый результат
        res_list = to_list(result.data)
        # сверяем результат с эталоном построчно
        for i in range(len(expected)):
            assert res_list[i] == pytest.approx(expected[i])

    # тест на защиту: нельзя делить на неквадратную матрицу
    def test_divide_non_square(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2, 3]]) # это не квадрат
        with pytest.raises(ValueError): # мы ждем, что программа выкинет ошибку ValueError
            m1.divide(m2)

    # самый важный тест: деление на «нулевую» (сингулярную) матрицу
    def test_divide_singular(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2], [2, 4]]) # у этой матрицы определитель = 0
        with pytest.raises(ValueError): # проверяем, что код выдает именно ValueError
            m1.divide(m2)

    # тест на некорректные данные при создании
    def test_init_uneven_rows(self, MatrixClass):
        with pytest.raises(ValueError): # ждем ошибку, если строки разной длины
            MatrixClass([[1, 2], [3]])
