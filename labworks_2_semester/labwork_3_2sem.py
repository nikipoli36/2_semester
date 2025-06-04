import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random

# Параметры исходной функции (произвольно выбранные)
a_true = 2.5
b_true = 1.8
c_true = 10.0


# 1. Генерация исходных данных
def generate_data(num_points=100):
    x_min = 0
    x_max = 5
    x = np.linspace(x_min, x_max, num_points)

    # Исходная функция без шума
    y_true = a_true * (b_true ** x) + c_true

    # Добавляем шум
    y = y_true + np.array([random.uniform(-3, 3) for _ in range(num_points)])

    return x, y, y_true


# 2. Функции для вычисления производных и MSE
def predict(x, a, b, c):
    """Предсказание значений y по параметрам модели"""
    return a * (b ** x) + c


def mse(y_true, y_pred):
    """Вычисление среднеквадратичной ошибки"""
    return np.mean((y_pred - y_true) ** 2)


def partial_a(x, y, a, b, c):
    """Частная производная по параметру a"""
    y_pred = predict(x, a, b, c)
    return (2 / len(x)) * np.sum((y_pred - y) * (b ** x))


def partial_b(x, y, a, b, c):
    """Частная производная по параметру b"""
    y_pred = predict(x, a, b, c)
    return (2 / len(x)) * np.sum((y_pred - y) * a * x * (b ** (x - 1)))


def partial_c(x, y, a, b, c):
    """Частная производная по параметру c"""
    y_pred = predict(x, a, b, c)
    return (2 / len(x)) * np.sum(y_pred - y)


# 3. Реализация градиентного спуска
def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    # Инициализация параметров (случайные начальные значения)
    a = random.uniform(0.1, 5.0)
    b = random.uniform(0.5, 3.0)
    c = random.uniform(-5.0, 15.0)

    # История параметров и ошибок для визуализации
    history = {
        'a': [a],
        'b': [b],
        'c': [c],
        'mse': [mse(y, predict(x, a, b, c))]
    }

    for epoch in range(epochs):
        # Вычисление градиентов
        grad_a = partial_a(x, y, a, b, c)
        grad_b = partial_b(x, y, a, b, c)
        grad_c = partial_c(x, y, a, b, c)

        # Обновление параметров
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c

        # Сохранение истории
        history['a'].append(a)
        history['b'].append(b)
        history['c'].append(c)
        history['mse'].append(mse(y, predict(x, a, b, c)))

    return history


# 4. Визуализация с ползунком
def visualize_results(x, y, y_true, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.25)

    # Первый график: данные и кривая регрессии
    ax1.scatter(x, y, alpha=0.7, label='Исходные данные с шумом')
    ax1.plot(x, y_true, 'g-', linewidth=2, label='Истинная функция')
    line, = ax1.plot(x, predict(x, history['a'][0], history['b'][0], history['c'][0]),
                     'r--', linewidth=2, label='Предсказание')
    ax1.set_title('Показательная регрессия: $y = a \cdot b^x + c$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.legend()

    # Второй график: изменение ошибки
    ax2.plot(history['mse'], 'b-')
    ax2.set_title('Изменение среднеквадратичной ошибки (MSE)')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('MSE')
    ax2.grid(True)
    ax2.set_yscale('log')  # Логарифмическая шкала для лучшей видимости

    # Создаем ось для ползунка
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Эпоха',
        valmin=0,
        valmax=len(history['mse']) - 1,
        valinit=0,
        valstep=1
    )

    # Функция обновления графиков
    def update(val):
        epoch = int(slider.val)
        a = history['a'][epoch]
        b = history['b'][epoch]
        c = history['c'][epoch]

        # Обновляем линию регрессии
        line.set_ydata(predict(x, a, b, c))

        # Обновляем заголовок с параметрами
        ax1.set_title(f'Показательная регрессия: $y = {a:.3f} \cdot {b:.3f}^x + {c:.3f}$ (Эпоха: {epoch})')

        # Обновляем точку на графике ошибок
        if epoch > 0:
            ax2.plot([epoch - 1, epoch], [history['mse'][epoch - 1], history['mse'][epoch]], 'r-')
        ax2.plot(epoch, history['mse'][epoch], 'ro')

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Кнопки для навигации
    ax_prev = plt.axes([0.25, 0.05, 0.1, 0.04])
    ax_next = plt.axes([0.65, 0.05, 0.1, 0.04])

    btn_prev = plt.Button(ax_prev, 'Предыдущая')
    btn_next = plt.Button(ax_next, 'Следующая')

    def prev_epoch(event):
        current = slider.val
        if current > slider.valmin:
            slider.set_val(current - 1)

    def next_epoch(event):
        current = slider.val
        if current < slider.valmax:
            slider.set_val(current + 1)

    btn_prev.on_clicked(prev_epoch)
    btn_next.on_clicked(next_epoch)

    plt.show()


# Главная функция
def main():
    # Генерация данных
    x, y, y_true = generate_data(100)

    # Градиентный спуск
    history = gradient_descent(x, y, learning_rate=0.01, epochs=1000)

    # Визуализация результатов
    visualize_results(x, y, y_true, history)

    # Вывод финальных параметров
    final_a = history['a'][-1]
    final_b = history['b'][-1]
    final_c = history['c'][-1]
    final_mse = history['mse'][-1]

    print("\nРезультаты обучения:")
    print(f"Истинные параметры: a = {a_true}, b = {b_true}, c = {c_true}")
    print(f"Найденные параметры: a = {final_a:.4f}, b = {final_b:.4f}, c = {final_c:.4f}")
    print(f"Среднеквадратичная ошибка (MSE): {final_mse:.4f}")


if __name__ == "__main__":
    main()