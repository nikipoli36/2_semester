import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try:

    df = pd.read_csv('датасет-1.csv', sep=';')


    if 'price' in df.columns:
        df['price'] = df['price'].str.replace(',', '.').astype(float)
    else:
        raise KeyError("Столбец 'price' не найден в датасете")


    plt.figure(figsize=(8, 6))
    plt.scatter(df['area'], df['price'], color='red')
    plt.title('Зависимость стоимости квартиры от площади')
    plt.xlabel('Площадь (кв.м.)')
    plt.ylabel('Стоимость (млн.руб)')
    plt.grid(True)
    plt.show()


    reg = LinearRegression()

    reg.fit(df[['area']], df['price'])


    pred_38 = reg.predict([[38]])
    print(f"\nПредсказанная стоимость квартиры 38 кв.м.: {pred_38[0]:.2f} млн.руб")

    pred_200 = reg.predict([[200]])
    print(f"Предсказанная стоимость квартиры 200 кв.м.: {pred_200[0]:.2f} млн.руб")

    df['predicted_price'] = reg.predict(df[['area']])
    print("\nСравнение фактических и предсказанных цен:")
    print(df[['area', 'price', 'predicted_price']].head())

    print("\nКоэффициенты модели:")
    print(f"a (коэффициент): {reg.coef_[0]:.4f}")
    print(f"b (интерсепт): {reg.intercept_:.4f}")
    print(f"Уравнение регрессии: price = {reg.coef_[0]:.4f} * area + {reg.intercept_:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(df['area'], df['price'], color='red', label='Фактические данные')
    plt.plot(df['area'], df['predicted_price'], 'b-', label='Линия регрессии')
    plt.title('Линейная регрессия: стоимость vs площадь')
    plt.xlabel('Площадь (кв.м.)')
    plt.ylabel('Стоимость (млн.руб)')
    plt.legend()
    plt.grid(True)
    plt.show()

    try:
        pred = pd.read_csv('prediction_price.csv', sep=';')

        pred['predicted_prices'] = reg.predict(pred[['area']])

        print("\nРезультаты прогнозирования:")
        print(pred)
        pred.to_excel('new.xlsx', index=False)
        print("\nРезультаты сохранены в файл 'new.xlsx'")

    except FileNotFoundError:
        print("\nОшибка: файл 'prediction_price.csv' не найден")
    except Exception as e:
        print(f"\nОшибка при обработке файла для прогнозирования: {str(e)}")

except FileNotFoundError:
    print("Ошибка: файл 'датасет.csv' не найден")
except KeyError as e:
    print(f"Ошибка: {str(e)}")
except Exception as e:
    print(f"Произошла ошибка: {str(e)}")