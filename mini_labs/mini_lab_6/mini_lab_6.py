import numpy as np
import matplotlib.pyplot as plt


def gradientDescend(
        func=lambda x: x ** 2,
        diffFunc=lambda x: 2 * x,
        x0=3.0,
        speed=0.01,
        epochs=100
):
    xList = [x0]
    yList = [func(x0)]

    x = x0
    for _ in range(epochs):
        x = x - speed * diffFunc(x)
        xList.append(x)
        yList.append(func(x))

    return xList, yList

# Функция и её производная
func = lambda x: np.exp(x) - 4 * x ** 2 + np.cos(x)
diffFunc = lambda x: np.exp(x) - 8 * x - np.sin(x)

# Параметры градиентного спуска
x0 = 3.0
speed = 0.01
epochs = 100
x_vals, y_vals = gradientDescend(func, diffFunc, x0, speed, epochs)

# Построение графика с центром в минимуме (x ≈ 0.5)
plt.figure(figsize=(12, 7))

# Диапазон с центром в минимуме
x_grid = np.linspace(-1, 2, 500)
y_grid = func(x_grid)

plt.plot(x_grid, y_grid, 'b-', linewidth=2, label="f(x) = $e^x - 4x^2 + \cos(x)$")
plt.plot(x_vals, y_vals, 'ro', alpha=0.7, label="Траектория градиентного спуска")
plt.plot(x_vals[-1], y_vals[-1], 'go', markersize=8, label="Конечная точка")

# Выделение минимума
plt.axvline(x=0.5, color='purple', linestyle='--', alpha=0.4, label="Ожидаемый минимум (x=0.5)")

plt.title("Метод градиентного спуска", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-1, 2)
plt.tight_layout()
plt.show()

# Проверка сходимости к минимуму
final_x = x_vals[-1]
print(f"Конечная точка: x = {final_x:.4f}, f(x) = {func(final_x):.4f}")
print(f"Расстояние до минимума: {abs(final_x - 0.5):.4f}")


# Поиск критического значения скорости
def find_critical_speed(func, diffFunc, x0=3.0, epochs=100, tol=0.1):
    low, high = 0.001, 0.1
    for _ in range(20):
        mid = (low + high) / 2
        x_vals, _ = gradientDescend(func, diffFunc, x0, mid, epochs)
        # Проверяем, сошлись ли мы к минимуму (x≈0.5)
        if abs(x_vals[-1] - 0.5) < tol:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 4)


critical_speed = find_critical_speed(func, diffFunc)
print(f"Граничное значение скорости: {critical_speed}")