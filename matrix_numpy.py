import numpy as np  # подключаем библиотеку нампай для быстрых расчетов

class Matrix:
    def __init__(self, data):
        # преобразуем входные данные в массив numpy
        self.data = np.array(data, dtype=float)  # превращаем обычный список в мощный массив нампай
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Все строки матрицы должны быть одинаковой длины")

        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]

    def __repr__(self):
        # округляем данные перед выводом для чистоты
        rounded_data = np.round(self.data, 10)
        return "\n".join(["[" + ", ".join(map(str, row)) + "]" for row in rounded_data])

    def transpose(self): # круто в 1 строчку да)
        # транспонирование через атрибут .T
        return Matrix(data=self.data.T)

    def multiply(self, other): # круто умножение через try except
        # матричное умножение через оператор @
        try:
            result_data = self.data @ other.data  # специальный символ для быстрого умножения матриц в нампай
            return Matrix(data=np.round(result_data, 4)) # в скобках сначала объект, потом число округления, у нас 4 ща
        except ValueError:
            raise ValueError(f"несовместимые размеры для умножения: {self.cols} != {other.rows}")

    def divide(self, other):
        # проверка на квадратную матрицу для инверсии матрицы потом ее умножения да все в два блока очень крутой np люблю
        if other.rows != other.cols:
            raise ValueError(f"делитель должен быть квадратным, получено: {other.rows}x{other.cols}")
        try:
            # умножение на обратную матрицу
            inverse_other = np.linalg.inv(other.data) # np.linalg.inv() тут ключевое, быстро инвертирует матрицу
            result_data = self.data @ inverse_other # Ну тут собака это умножение а деление матриц это умножение на инвертир матрицу
            return Matrix(data=np.round(result_data, 4)) #  округляем до 4х знаков чтобы не захламлять вывод/логи и тд
        except np.linalg.LinAlgError: # если не сработало, то по одной причине:
            raise ValueError(f"матрица с данными {other.data} вырождена, деление невозможно")



# ОГРОМНЫЙ БЛОК ДЕМОНСТРАЦИИ КОДА
   # --- ТРАНСПОНИЗАЦИЯ ---
M = Matrix([[1,2,3],[1,2,3]])
if __name__ == "__main__":
    print("--- Кейс 0: Матрица транспонированная на обычную")
    m_t = M.transpose()
    try:
        print(f"Матрица A ({M.rows}x{M.cols}):\n{M}")
        print(f"Матрица B ({m_t.rows}x{m_t.cols}):\n{m_t}")
        # стык: 2x3 и 3x2 -> 3 == 3 значит можно
        result = M.multiply(m_t)
        print("\nРезультат (можно умножить, так как на стыке 3 и 3):")
        print(result)
    except ValueError as e:
        print(f"ошибка: {e}")
    
        # --- ДРУГОЙ БЛОК ---
    print("Главное правило: внутренние размеры на стыке матриц должны быть одинаковыми.\n")
    m_3x3 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # m_3x3 удобное сокращение матрицы 3х3 т.е. квадратная
    print("--- Кейс 1: Матрица 2x3 на 3x3 ---")
    m_2x3 = Matrix([[1, 2, 3], [1, 2, 3]]) # матрица две строки 3 символа в каждой то есть три столбца
    print(f"Матрица A (2x3):\n{m_2x3}")
    print(f"Матрица B (3x3):\n{m_3x3}")
    try:
        # стык: 2x3 и 3x3 -> 3 == 3 значит можно
        result = m_2x3.multiply(m_3x3)
        print("\nРезультат (можно умножить, так как на стыке 3 и 3):")
        print(result)
    except ValueError as e:
        print(f"ошибка: {e}")
    
    print("\n--- Кейс 2: Матрица 3x2 на 3x3 ---")
    m_3x2 = Matrix([[1, 2], [1, 2], [1, 2]]) # сокращение матрицы три строки в каждой 2 столбца
    print(f"Матрица A (3x2):\n{m_3x2}")
    print(f"Матрица B (3x3):\n{m_3x3}")
    try:
        # стык: 3x2 и 3x3 -> 2 != 3, значит нельзя!
        result2 = m_3x2.multiply(m_3x3)
        print(result2)
    except ValueError as e:
        print(f"ошибка: {e} \n")
    
    # --- КЕЙС 3 - ДЕЛЕНЕИ МАТРИЦ ---
    # (делимое)
    # 1. Создаем матрицы
    A_d = Matrix([[2.0, 5.0], [3.0, 4.0]])
    
    B_d = Matrix([[1.0, 2.0], [3.0, 4.0]])
    
    # 2. Делим их друг на друга внутри класса
    C_d = A_d.divide(B_d)
    
    # 3. Просто печатаем объект C! Python сам вызовет метод __repr__
    print(f"Деление матриц:\nМатрица 1:\n{A_d}\nМатрица 2:\n{B_d}\nРезультат деления матриц:\n{C_d}")
    
    
    
    
