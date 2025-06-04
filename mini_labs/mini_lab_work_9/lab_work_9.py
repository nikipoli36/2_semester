import random

random.seed(91822)
a = random.randint(1, 20)
print("Номер задания >> ",a)

"""
Найти среднее арифметическое значение элементов массива, 
расположенных между минимальным и максимальным элементами массива
"""

def average_mx(arr):
    if not arr:        return None
    min_ind = arr.index(min(arr))
    max_ind = arr.index(max(arr))

    start = min(min_ind, max_ind) + 1
    end = max(min_ind, max_ind)

    sub_arr = arr[start:end]

    if not sub_arr:        return None
    average = sum(sub_arr) / len(sub_arr)
    return average
try:
    with open('C:/Users/CompLife33/PycharmProjects/pythonProject/venv/lab_work_9/labs_9_in.txt', 'r') as file:
        arr = list(map(int, file.read().split()))
except FileNotFoundError:
    print("Файл input.txt не найден.")
    exit()
# Вычисляем среднее арифметическое
result = average_mx(arr)

# Запись результата в файл output.txt
with open('labs_9_out.txt', 'w') as file:
    if result is not None:
        file.write(f"Среднее арифметическое значение >> {result}\n")
    else:
        file.write("Нет элементов между минимальным и максимальным значениями.\n")

print("Результат записан в файл 'labs_9_out.txt'.\n")
print(arr)

result = average_mx(arr)

if result is not None:
    print("Среднее арифметическое значение >>", result)
else:
    print("Нет элементов между минимальным и максимальным значениями.")