import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import random


#  Генерация исходной функции
def original_function(x):
    return np.sin(x) * x ** 2 + 2 * x + 5 * np.cos(x)


#  Генерация данных с шумом
x_min = -5
x_max = 5
num_points = 100

# Генерация равномерно распределенных точек по X
x = np.linspace(x_min, x_max, num_points)

# Генерация значений Y с шумом
y_noise = np.array([random.uniform(-1.5, 1.5) for _ in range(num_points)])
y = original_function(x) + y_noise

#  Подготовка данных для обучения (преобразование в 2D-массив)
X = x.reshape(-1, 1)

#  Создание и обучение моделей регрессии
models = [
    ("Support Vector Regression", SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)),
    ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]

# Создание фигуры для визуализации
plt.figure(figsize=(18, 12))
plot_idx = 1

# Список для хранения MSE
mse_results = []

#  Обучение моделей и визуализация результатов
for name, model in models:
    # Обучение модели
    model.fit(X, y)

    # Прогнозирование
    y_pred = model.predict(X)

    # Расчет MSE
    mse = mean_squared_error(y, y_pred)
    mse_results.append((name, mse))

    # Построение графика
    plt.subplot(2, 2, plot_idx)

    # Исходные точки с шумом
    plt.scatter(x, y, color='blue', alpha=0.6, label='Исходные точки')

    # Исходная функция
    plt.plot(x, original_function(x), 'g-', linewidth=2, label='Исходная функция')

    # Предсказанная функция
    plt.plot(x, y_pred, 'r-', linewidth=2, label=f'Предсказание ({name})')

    plt.title(f'{name}\nMSE: {mse:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(alpha=0.3)

    plot_idx += 1

#  Сравнение результатов по MSE
plt.subplot(2, 2, 4)
models_names = [m[0] for m in mse_results]
mse_values = [m[1] for m in mse_results]

bars = plt.bar(models_names, mse_values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Сравнение среднеквадратичной ошибки (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=15)

# Добавление значений на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('regression_results.png', bbox_inches='tight')
plt.show()

print("\nРезультаты сравнения методов:")
for name, mse in mse_results:
    print(f"{name}: MSE = {mse:.4f}")

best_model = min(mse_results, key=lambda x: x[1])
print(f"\nЛучший результат: {best_model[0]} (MSE = {best_model[1]:.4f})")