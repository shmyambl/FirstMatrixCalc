
import pytest  # Библиотека для удобного запуска тестов
import numpy as np  # Библиотека для математических операций

# Импортируем классы матриц: один на чистом Python, второй на NumPy
from matrix_scratch import Matrix as ScratchMatrix
from matrix_numpy import Matrix as NumpyMatrix

# Вспомогательная функция: превращает любые данные (NumPy или списки) в обычный список списков
def to_list(data):
    if hasattr(data, 'tolist'): return data.tolist() # Если это объект NumPy, используем его метод tolist()
    return [list(row) for row in data] # Иначе просто перебираем строки

# Параметризация: этот декоратор заставляет pytest запустить все тесты ниже ДВАЖДЫ
# (сначала для ScratchMatrix, потом для NumpyMatrix)
@pytest.mark.parametrize("MatrixClass", [ScratchMatrix, NumpyMatrix])
class TestMatrixOperations:

    # Тест на транспонирование (поворот матрицы)
    def test_transpose(self, MatrixClass):
        matrix = MatrixClass([[1, 2], [3, 4]]) # Создаем матрицу
        result = matrix.transpose() # Вызываем поворот
        assert to_list(result.data) == [[1, 3], [2, 4]] # Проверяем, что строки стали столбцами

    # Тест на обычное деление матриц
    def test_divide_compatible(self, MatrixClass):
        m1 = MatrixClass([[2, 5], [3, 4]]) # Делимое
        m2 = MatrixClass([[1, 2], [3, 4]]) # Делитель
        result = m1.divide(m2) # Выполняем деление
        expected = [[3.5, -0.5], [0.0, 1.0]] # Ожидаемый результат (посчитан заранее)
        # approx нужен, так как компьютеры могут давать мизерные погрешности в дробях
        assert to_list(result.data) == pytest.approx(expected)

    # Тест на защиту: нельзя делить на матрицу, которая не квадратная
    def test_divide_non_square(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2, 3]]) # Это не квадрат
        with pytest.raises(ValueError): # Мы ЖДЕМ, что программа выкинет ошибку ValueError
            m1.divide(m2)

    # Самый важный тест: деление на «нулевую» (сингулярную) матрицу
    def test_divide_singular(self, MatrixClass):
        m1 = MatrixClass([[1, 2], [3, 4]])
        m2 = MatrixClass([[1, 2], [2, 4]]) # У этой матрицы определитель = 0, делить нельзя
        with pytest.raises(ValueError): # Проверяем, что код выдает именно ValueError
            m1.divide(m2)

    # Тест на кривые данные при создании
    def test_init_uneven_rows(self, MatrixClass):
        with pytest.raises(ValueError): # Ждем ошибку, если в одной строке 2 числа, а в другой 1
            MatrixClass([[1, 2], [3]])
