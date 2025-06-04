import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
import math

def generate_data(k=3, n=20):
    data_x = []
    data_y = []
    centers = []
    radii = []
    labels = []
    # Генерируем центры и радиусы для каждого кластера
    min_distance = 5.0
    max_radius = 2.0

    for i in range(k):
        valid_position = False
        while not valid_position:
            cx = random.uniform(0, 20)
            cy = random.uniform(0, 20)
            r = random.uniform(1.0, max_radius)

            valid_position = True
            for j in range(len(centers)):
                dist = math.sqrt((cx - centers[j][0]) ** 2 + (cy - centers[j][1]) ** 2)
                if dist < (r + radii[j] + min_distance):
                    valid_position = False
                    break

            if valid_position:
                centers.append((cx, cy))
                radii.append(r)

                # Генерируем точки внутри кольца
                for _ in range(n):
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, r)
                    x = cx + distance * math.cos(angle)
                    y = cy + distance * math.sin(angle)

                    data_x.append(x)
                    data_y.append(y)
                    labels.append(i)

    return np.array(data_x), np.array(data_y), np.array(labels), centers, radii

#  Реализация метода k-средних
def k_means(x, y, k, student_id=1):
    # Выбор критерия остановки по номеру студента
    stop_criterion = student_id % 3

    # Инициализация центроидов случайными точками
    indices = np.random.choice(len(x), k, replace=False)
    centroids = np.column_stack((x[indices], y[indices]))

    # Для хранения истории изменений
    history_centroids = [centroids.copy()]
    history_labels = []

    # Основной цикл алгоритма
    max_iterations = 100
    tolerance = 0.001
    prev_labels = None

    for iteration in range(max_iterations):
        # Расстояния от точек до центроидов
        distances = np.sqrt((x[:, np.newaxis] - centroids[:, 0]) ** 2 +
                            (y[:, np.newaxis] - centroids[:, 1]) ** 2)
        labels = np.argmin(distances, axis=1)
        history_labels.append(labels.copy())

        # Пересчет центроидов
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = np.column_stack((x[labels == i], y[labels == i]))
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        # Проверка критериев остановки
        if stop_criterion == 0 and iteration >= 99:
            break
        elif stop_criterion == 1:
            centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))
            if np.max(centroid_shift) < tolerance:
                break
        elif stop_criterion == 2 and prev_labels is not None and np.array_equal(labels, prev_labels):
            break

        prev_labels = labels.copy()
        centroids = new_centroids.copy()
        history_centroids.append(centroids.copy())

    return history_centroids, history_labels

def main(student_id=1):
    # Генерация данных
    k = 3
    x, y, true_labels, centers, radii = generate_data(k=k)

    history_centroids, history_labels = k_means(x, y, k, student_id)

    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25)

    # Создаем ось для ползунка
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Эпоха',
        valmin=0,
        valmax=len(history_labels) - 1,
        valinit=0,
        valstep=1
    )
    # Начальные цвета для кластеров
    colors = plt.cm.get_cmap('viridis', k)

    # Функция для отрисовки состояния
    def plot_state(epoch):
        ax.clear()

        # Получаем данные для выбранной эпохи
        labels = history_labels[epoch]
        centroids = history_centroids[epoch]

        # Отображаем точки
        for i in range(k):
            cluster_points = np.column_stack((x[labels == i], y[labels == i]))
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           c=[colors(i)] * len(cluster_points), s=30, alpha=0.6,
                           label=f'Кластер {i + 1}')

        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Центроиды')

        # Настройка графика
        ax.set_title(f'Метод k-средних (Эпоха: {epoch})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(min(x) - 1, max(x) + 1)
        ax.set_ylim(min(y) - 1, max(y) + 1)
        fig.canvas.draw_idle()

    # Инициализируем начальное состояние
    plot_state(0)

    def update(val):
        epoch = int(slider.val)
        plot_state(epoch)

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
if __name__ == "__main__":
    main(student_id=22)