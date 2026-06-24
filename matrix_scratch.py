class Matrix:
    def __init__(self, data):
        # проверяем, чтобы все строки были одинаковой длины
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Все строки матрицы должны быть одинаковой длины")
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

    def __repr__(self):
        return "\n".join(["[" + ", ".join(map(str, row)) + "]" for row in self.data])


    def transpose(self):
        new_data = [[0 for _ in range(self.rows)] for _ in range(self.cols)]
        for i in range(self.cols):
          for j in range(self.rows):
            new_data[i][j] = self.data[j][i]
        return Matrix(data=new_data)



    def multiply(self,other):
      if self.cols != other.rows:
        raise ValueError("Несовместимые матрицы")
      # важно что в у множении новый матрчиный объект будет в длинне cols  второго множителя, а в числу rows по первому
      new_data = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
      for i in range(self.rows):
        for j in range(other.cols):
          for k in range(self.cols):
            new_data[i][j] += self.data[i][k] * other.data[k][j]
      return Matrix(data=new_data)


    def divide(self, other): # поскольку без Numpy Это очень трудно делать, я просто скопирую код из интернета и постараюсь его понять и скомментировать и поменять названия на свои, но с numpy я бы сам написал, изначально пытался поделить через код стандартного умножения, но возвести всю матрицу в отриц степень оказывается трудно

        # n был бы len, но у матриц это .rows
        n = other.rows
        if other.rows != other.cols: # Базовая проверка квадратичности матрицы
            raise ValueError("Для деления матрица-делитель должна быть квадратной")

        #  НАХОЖДЕНИЕ ОБРАТНОЙ МАТРИЦЫ (Матрица умноженная на -1 степень) ДЛЯ 'other' МЕТОДОМ гаусса-жордана

        # Создаем единичную матрицу размера n x n то есть квадратичную
        temporary_for_divide = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        # Создаем расширенную матрицу: склеиваем строки other.data и единичной матрицы
        augmented = [other.data[i] + temporary_for_divide[i] for i in range(n)] # Augmented matrix — это матрица, полученная путем объединения двух других матриц. Ее формулы есть в заметках у меня

        # Прямой и обратный ход метода Гаусса
        for i in range(n):
            # Ищем строку с ненулевым элементом в текущем столбце
            pivot_row = i # pivot - это главный (опорный) элемент, который выбирается на текущем шаге алгоритма для проведения вычислений
            while pivot_row < n and abs(augmented[pivot_row][i]) < 1e-9: # в этом цикле программа перескакивает по строкам чтобы найти определитель не равный нулю, есть ограничение по кол-ву итераций
                pivot_row += 1
            if pivot_row == n: # если определитель равен нулю то есть детерминант
                raise ValueError(
                    "Матрица 'other' вырождена. Ее нельзя обратить и поделить на нее!"
                )
            # Меняем строки местами, если главный элемент не на диагонали
            if pivot_row != i:
                augmented[i], augmented[pivot_row] = ( # если в опорной строке != нашему инднексу, то транспонируем нашу расширенную матрицу
                    augmented[pivot_row],
                    augmented[i],
                )
            # Делим текущую строку на диагональный элемент, чтобы получить 1
            pivot = augmented[i][i]
            augmented[i] = [x / pivot for x in augmented[i]]
            # Зануляем элементы в текущем столбце для всех остальных строк
            for j in range(n):
                if i != j:
                    factor = augmented[j][i] # 1. Факторизация матриц (Matrix Factorization) Это метод разложения сложной исходной матрицы на произведение двух или более более простых матриц-множителей (факторов)
                    augmented[j] = [ #  по сути факторизация это метод сжатия данных
                        augmented[j][k] - factor * augmented[i][k] # умножаем все на 0 => все == 0
                        for k in range(2 * n)
                    ]
        # Вырезаем правую часть расширенной матрицы — это и есть inverse_other
        inverse_other = [row[n:] for row in augmented] # используем срезы, которые узнали от массивов.

        # УМНОЖЕНИЕ МАТРИЦЫ 'self' НА 'inverse_other'

        rows_self, cols_self = self.rows, self.cols # присваиваем нужные значения
        rows_other_inv, cols_other_inv = len(inverse_other), len(inverse_other[0])

        if cols_self != rows_other_inv:
            raise ValueError(
                "Размеры матриц 'self' и 'other' не совпадают для умножения!"
            )

        # Создаем пустую матрицу для результата
        result_data = [[0.0 for _ in range(cols_other_inv)] for _ in range(rows_self)]

        # просто умножение матриц) внутри деления
        for i in range(rows_self):
            for j in range(cols_other_inv):
                for k in range(cols_self):
                    result_data[i][j] += self.data[i][k] * inverse_other[k][j]
        return Matrix(data=result_data)






   # --- ТРАНСПОНИЗАЦИЯ ---
M = Matrix([[1,2,3],[1,2,3]])
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




