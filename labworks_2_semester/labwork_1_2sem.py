import random
import math
import matplotlib.pyplot as plt

def generate_data(pointsCount1=50, pointsCount2=50,
                  xMin1=0, xMax1=5, yMin1=0, yMax1=5,
                  xMin2=3, xMax2=8, yMin2=3, yMax2=8):
    x = []
    y = []
    # Генерация точек для класса 0
    for _ in range(pointsCount1):
        x.append([random.uniform(xMin1, xMax1), random.uniform(yMin1, yMax1)])
        y.append(0)
    # Генерация точек для класса 1
    for _ in range(pointsCount2):
        x.append([random.uniform(xMin2, xMax2), random.uniform(yMin2, yMax2)])
        y.append(1)
    return x, y

def train_test_split(x, y, p=0.8):
    combined = list(zip(x, y))
    random.shuffle(combined)
    split_index = int(len(combined) * p)
    train = combined[:split_index]
    test = combined[split_index:]

    x_train = [item[0] for item in train]
    y_train = [item[1] for item in train]
    x_test = [item[0] for item in test]
    y_test = [item[1] for item in test]

    return x_train, x_test, y_train, y_test

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def fit(x_train, y_train, x_test, k=3):
    y_predict = []

    for test_point in x_test:
        distances = []
        for i, train_point in enumerate(x_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        class_votes = {}
        for _, class_label in neighbors:
            class_votes[class_label] = class_votes.get(class_label, 0) + 1

        predicted_class = max(class_votes, key=class_votes.get)
        y_predict.append(predicted_class)
    return y_predict

def compute_accuracy(y_test, y_predict):
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            correct += 1
    return correct / len(y_test)

# Визуализация результатов
def visualize(x_train, y_train, x_test, y_test, y_predict):
    plt.figure(figsize=(10, 8))

    # Отображение обучающих точек
    for i, point in enumerate(x_train):
        if y_train[i] == 0:
            plt.plot(point[0], point[1], 'bo')
        else:
            plt.plot(point[0], point[1], 'bx')

    # Отображение тестовых точек
    for i, point in enumerate(x_test):
        if y_test[i] == y_predict[i]:  # Правильная классификация
            if y_test[i] == 0:
                plt.plot(point[0], point[1], 'go')
            else:
                plt.plot(point[0], point[1], 'gx')
        else:  # Неправильная классификация
            if y_test[i] == 0:
                plt.plot(point[0], point[1], 'ro')
            else:
                plt.plot(point[0], point[1], 'rx')

# График
    plt.title('K Nearest Neighbors Classification')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Train Class 0'),
        plt.Line2D([0], [0], marker='x', color='blue', linestyle='None', markersize=10, label='Train Class 1'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct Class 0'),
        plt.Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=10, label='Correct Class 1'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Wrong Class 0'),
        plt.Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10, label='Wrong Class 1')
    ]
    plt.legend(handles=handles)

    plt.show()

x, y = generate_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, p=0.8)
y_predict = fit(x_train, y_train, x_test, k=3)

accuracy = compute_accuracy(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")

visualize(x_train, y_train, x_test, y_test, y_predict)