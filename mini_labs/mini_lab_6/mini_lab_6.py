import numpy as np
import matplotlib.pyplot as plt


# Генерация функции и её производной
def generate_function():
    """
    Генерируем дифференцируемую трансцендентно-алгебраическую функцию
    f(x) = x^2 + 5*sin(x) + e^(-x^2)
    Производная: f'(x) = 2x + 5*cos(x) - 2x*e^(-x^2)
    Минимум: примерно в точке x ≈ -1.5 (локальный минимум)
    """
    func = lambda x: x ** 2 + 5 * np.sin(x) + np.exp(-x ** 2)
    diff_func = lambda x: 2 * x + 5 * np.cos(x) - 2 * x * np.exp(-x ** 2)
    return func, diff_func


# Реализация метода градиентного спуска
def gradientDescend(func=None, diffFunc=None, x0=3, speed=0.01, epochs=100):
    if func is None:
        func = lambda x: x ** 2
    if diffFunc is None:
        diffFunc = lambda x: 2 * x

    x_list = [x0]
    y_list = [func(x0)]

    x_current = x0
    for _ in range(epochs):
        x_current = x_current - speed * diffFunc(x_current)
        x_list.append(x_current)
        y_list.append(func(x_current))

    return x_list, y_list


if __name__ == "__main__":
    # Генерация функции
    func, diffFunc = generate_function()

    # Параметры градиентного спуска
    x0 = 3
    speed = 0.1
    epochs = 100

    # Запуск градиентного спуска
    x_list, y_list = gradientDescend(func, diffFunc, x0, speed, epochs)

    # Визуализация результатов
    x_vals = np.linspace(-4, 4, 400)
    y_vals = func(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', label='Функция: $x^2 + 5\sin(x) + e^{-x^2}$')
    plt.plot(x_list, y_list, 'ro-', alpha=0.6, label='Траектория градиентного спуска')
    plt.plot(x_list[-1], y_list[-1], 'go', markersize=8, label='Финальная точка')
    plt.title('Градиентный спуск')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Анализ сходимости
    final_x = round(x_list[-1], 2)
    final_y = round(y_list[-1], 2)
    print(f"Стартовая точка: x = {x0}")
    print(f"Финальная точка: x = {final_x}, y = {final_y}")

    # Поиск критического значения скорости обучения
    critical_speed = None
    speeds = np.linspace(0.1, 0.5, 50)

    for test_speed in speeds:
        test_x, _ = gradientDescend(func, diffFunc, x0, test_speed, epochs)
        if any(np.abs(x) > 100 for x in test_x):  # Критерий расходимости
            critical_speed = test_speed
            break

    if critical_speed is not None:
        print(f"\nКритическое значение скорости: {critical_speed:.3f}")
        print(f"- При speed < {critical_speed:.3f} метод сходится")
        print(f"- При speed > {critical_speed:.3f} метод расходится")
    else:
        print("\nМетод сходится при всех тестируемых скоростях")

# Дополнительный анализ сходимости
print("\nПример поведения при разных скоростях:")
for test_speed in [0.15, critical_speed]:
    test_x, test_y = gradientDescend(func, diffFunc, x0, test_speed, 50)
    status = "сходится" if abs(test_x[-1]) < 10 else "расходится"
    print(f"speed = {test_speed:.3f}: конечная точка x = {test_x[-1]:.2f} ({status})")