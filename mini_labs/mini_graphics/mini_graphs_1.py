"""
Вариант №22
"""

import numpy as np
import matplotlib.pyplot as plt

print("Вы сможете построить график функции: y = a * x**3 - np.sin(b * x)")
a = int(input("Задайте параметр 'a' >> "))
b = int(input("Задайте параметр 'b' >> "))
n = int(input("Задайте размерность массива >> "))

x = np.linspace(0, n, 100)

y = a * x**3 - np.sin(b * x)

plt.figure(figsize=(10, 8))
plt.plot(x, y, label=f'y = {a}*x^3 - sin({b}*x)')
plt.title('График функции y = a*x^3 - sin(b*x)')
plt.xlabel('x')
plt.yticks(rotation = 90)
plt.ylabel('y')
plt.grid(True)
plt.show()