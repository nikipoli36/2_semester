"""
Вариант №22
"""
import matplotlib.pyplot as plt

with open('data.txt', 'r') as file:
    lines = file.readlines()
    x = list(map(float, lines[2].strip().split()))
    y = list(map(float, lines[3].strip().split()))

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("График данных из файла")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()