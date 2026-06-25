import pytest  # подключаем библиотеку для удобного запуска проверок
import numpy as np  # берем нампай для работы с массивами

# импортируем ваши классы матриц из созданных файлов
from matrix_scratch import Matrix as ScratchMatrix
from matrix_numpy import Matrix as NumpyMatrix

# функция-помощник: переводит любые данные в обычные списки питона для сравнения
def to_list(data):
    if hasattr(data, 'tolist'):  # если это массив numpy, используем его встроенный метод
        return data.tolist()
    # если это список, убеждаемся, что вложенные строки тоже стали списками
    return [list(row) if isinstance(row, (list, tuple, np.ndarray)) else row for row in data]

# декоратор заставляет pytest прогнать все тесты ниже дважды: для обычной матрицы и для numpy
@pytest.mark.parametrize("MatrixClass", [ScratchMatrix, NumpyMatrix])
class TestMatrixOperations:

    def test_transpose(self, MatrixClass):
        data = [[1, 2, 3], [1, 2, 3]]  # берем исходную матрицу
        expected = [[1, 1], [2, 2], [3, 3]]  # так она должна выглядеть после поворота
        matrix = MatrixClass(data)  # создаем объект
        result = matrix.transpose()  # переворачиваем
        assert to_list(result.data) == expected  # сверяем результат с ожиданием

    def test_multiply_compatible(self, MatrixClass):
        a = [[1, 2, 3], [1, 2, 3]]  # первая матрица
        b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # вторая матрица
        expected = [[30, 36, 42], [30, 36, 42]]  # результат умножения
        matrix_a = MatrixClass(a)  # создаем первую
        matrix_b = MatrixClass(b)  # создаем вторую
        result = matrix_a.multiply(matrix_b)  # перемножаем
        assert to_list(result.data) == expected  # проверяем ответ

    def test_multiply_incompatible(self, MatrixClass):
        a = [[1, 2], [3, 4], [5, 6]]  # узкая матрица
        b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # широкая матрица (размеры не подходят)
        matrix_a = MatrixClass(a)
        matrix_b = MatrixClass(b)
        with pytest.raises(ValueError):  # ожидаем, что программа выдаст ошибку ValueError
            matrix_a.multiply(matrix_b)

    def test_divide_compatible(self, MatrixClass):
        m1 = [[2, 5], [3, 4]]  # что делим
        m2 = [[1, 2], [3, 4]]  # на что делим
        expected = [[3.5, -0.5], [0.0, 1.0]]  # правильный ответ
        matrix_1 = MatrixClass(m1)
        matrix_2 = MatrixClass(m2)
        result = matrix_1.divide(matrix_2)  # делим

        res_list = to_list(result.data)
        for i in range(len(expected)):
            # сверяем дробные числа через approx (с учетом мелких погрешностей)
            assert res_list[i] == pytest.approx(expected[i])

    def test_divide_non_square(self, MatrixClass):
        m1 = [[1, 2], [3, 4]]
        m2 = [[1, 2, 3], [4, 5, 6]]  # нельзя делить на неквадратную матрицу
        matrix_1 = MatrixClass(m1)
        matrix_2 = MatrixClass(m2)
        with pytest.raises(ValueError):  # проверяем, сработает ли защита
            matrix_1.divide(matrix_2)

    def test_divide_singular(self, MatrixClass):
        m1 = [[1, 2], [3, 4]]
        m2 = [[1, 2], [2, 4]]  # у этой матрицы определитель ноль (делить нельзя)
        matrix_1 = MatrixClass(m1)
        matrix_2 = MatrixClass(m2)
        # требуем строго ошибку ValueError для обеих реализаций
        with pytest.raises(ValueError):
            matrix_1.divide(matrix_2)

    def test_init_uneven_rows(self, MatrixClass):
        data = [[1, 2, 3], [4, 5]]  # матрица с разной длиной строк (ошибка)
        with pytest.raises(ValueError):  # убеждаемся, что такая матрица не создастся
            MatrixClass(data)
