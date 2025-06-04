import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Генерация 5 вариантов исходных данных
def generate_datasets():
    seed = 30
    n_samples = 500

    # 1. Две окружности
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)

    # 2. Две параболы
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

    # 3. Хаотичное распределение
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 0.5], random_state=seed, centers=2)

    # 4. Точки вокруг прямых
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed, centers=2)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    x_aniso = np.dot(blobs[0], transformation)
    aniso = (x_aniso, blobs[1])

    # 5. Слабо пересекающиеся области
    separated = datasets.make_blobs(n_samples=n_samples, random_state=seed + 1, centers=2, cluster_std=1.2)

    return [
        ("Две окружности", noisy_circles),
        ("Две параболы", noisy_moons),
        ("Хаотичное распределение", varied),
        ("Точки вокруг прямых", aniso),
        ("Слабо пересекающиеся области", separated)
    ]


# Создание моделей классификации
def create_models():
    models = [
        ("K-ближайших соседей", KNeighborsClassifier(n_neighbors=5)),
        ("Логистическая регрессия", LogisticRegression(max_iter=1000)),
        ("Многослойный перцептрон", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True
        ))
    ]
    return models


# Функция для визуализации результатов
def plot_decision_boundary(ax, model, X, y, title):
    # Создание сетки для построения границы
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Прогнозирование для всех точек сетки
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Построение контурной карты
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')

    # Отображение обучающих точек
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='coolwarm', edgecolors='k')

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# Основной код
datasets_list = generate_datasets()
models = create_models()

# Создание фигуры для визуализации
fig, axes = plt.subplots(len(datasets_list), len(models), figsize=(18, 25))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# Обработка каждого набора данных
for ds_idx, (ds_name, (X, y)) in enumerate(datasets_list):
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение и оценка каждой модели
    for model_idx, (model_name, model) in enumerate(models):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Визуализация
        ax = axes[ds_idx, model_idx]
        plot_decision_boundary(ax, model, X, y, f"{model_name}\n{ds_name}\nТочность: {accuracy:.2f}")

plt.savefig('classification_results.png', bbox_inches='tight')
plt.show()