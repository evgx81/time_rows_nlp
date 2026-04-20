import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def calculate_correlation(x: np.ndarray, y: np.ndarray, min_lag=0, max_lag=10):
    """
    Вычисляет взаимокорреляцию между x и y для каждого сдвига (лага).

    Параметры:
        x, y    — входные временные ряды
        min_lag — минимальный сдвиг
        max_lag — максимальный сдвиг

    Возвращает:
        lags  — массив сдвигов
        corrs — массив коэффициентов корреляции
    """
    n = len(x)
    lags = range(min_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        # Сдвигаем ряды относительно друг друга в зависимости от знака лага
        if lag < 0:
            # Отрицательный сдвиг: x смещён вправо относительно y
            x_shifted = x[-lag:]
            y_shifted = y[: n + lag]
        elif lag > 0:
            # Положительный сдвиг: y смещён вправо относительно x
            x_shifted = x[: n - lag]
            y_shifted = y[lag:]
        else:
            # Нулевой сдвиг: ряды не смещены
            x_shifted = x
            y_shifted = y

        # Считаем корреляцию только если точек достаточно
        if len(x_shifted) > 2:
            r = np.corrcoef(x_shifted, y_shifted)[0, 1]
            corrs.append(r if not np.isnan(r) else 0)

    return np.array(lags), np.array(corrs)


# Вспомогательная функция: безопасно получить значение ряда по индексу
# Если индекс выходит за границы — берём крайнее значение (клиппинг)
def get_value(series, idx):
    if idx < 0:
        return series[0]
    elif idx >= len(series):
        return series[-1]
    else:
        return series[idx]


def build_dataset_for_nlp(x1, x2, x3, Y, tau1, tau2, tau3, lam, l):
    """
    Формирует обучающую выборку для нейронной сети из временных рядов.

    Параметры:
        x1, x2, x3 — входные временные ряды
        Y           — целевой временной ряд
        tau1..tau3  — временные сдвиги для каждого канала (из корреляционного анализа)
        lam         — общая задержка (максимальный сдвиг)
        l           — количество точек предыстории для каждого канала

    Возвращает:
        X — матрица признаков
        Y — вектор целевых значений
    """

    X_list = []
    Y_list = []
    min_idx = lam + max(0, -tau1, -tau2, -tau3) + l
    max_idx = len(Y) - 1

    for t in range(min_idx, max_idx):
        row = []

        # Добавляем l точек предыстории для канала x1
        for d in range(-l + 1, 1):
            idx = t - lam + tau1 + d
            row.append(get_value(x1, idx))

        # Добавляем l точек предыстории для канала x2
        for d in range(-l + 1, 1):
            idx = t - lam + tau2 + d
            row.append(get_value(x2, idx))

        # Добавляем l точек предыстории для канала x3
        for d in range(-l + 1, 1):
            idx = t - lam + tau3 + d
            row.append(get_value(x3, idx))

        X_list.append(row)
        Y_list.append(Y[t])

    return np.array(X_list), np.array(Y_list)


if __name__ == "__main__":

    # Получаем данные множественного временного ряда из Excel-файла
    data_filename = "Данные_к_лаб_4.xlsx"
    df = pd.read_excel(data_filename)

    df = df.iloc[:400].copy()

    x1 = df["x1"].values
    x2 = df["x2"].values
    x3 = df["x3"].values
    y = df["Y"].values

    print(f"Количество наблюдений: {len(y)}")
    print(f"Значения Y изменяются в интервале [{y.min():.2f}, {y.max():.2f}]")

    # Рассчитаем коэффициенты парной корреляции для выявления необходимого смещения для формирования обучающей выборки

    # Вычисляем временные сдвиги
    l_min = 0
    l_max = 50

    # Считаем корреляции для каждого входа
    lags1, corr1 = calculate_correlation(x1, y, l_min, l_max)
    lags2, corr2 = calculate_correlation(x2, y, l_min, l_max)
    lags3, corr3 = calculate_correlation(x3, y, l_min, l_max)

    # Определяем сдвиги по временным каналам

    # Находим сдвиги
    tau1_idx = np.argmax(np.abs(corr1))
    tau2_idx = np.argmax(np.abs(corr2))
    tau3_idx = np.argmax(np.abs(corr3))

    tau1 = lags1[tau1_idx]
    tau2 = lags2[tau2_idx]
    tau3 = lags3[tau3_idx]

    print("Результаты корреляционного анализа:")
    print(
        f"Канал: x1 - Y: максимальная корреляция равна {np.max(np.abs(corr1)):.2f} при сдвиге {tau1}"
    )
    print(
        f"Канал: x2 - Y: максимальная корреляция равна {np.max(np.abs(corr2)):.2f} при сдвиге {tau2}"
    )
    print(
        f"Канал: x3 - Y: максимальная корреляция равна {np.max(np.abs(corr3)):.2f} при сдвиге {tau3}"
    )

    # Строим графики взаимокорреляционных функций
    for corr, tau, lag, label in zip(
        [corr1, corr2, corr3],
        [tau1, tau2, tau3],
        [lags1, lags2, lags3],
        ["X1", "X2", "X3"],
    ):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(lag, corr, color="blue")
        ax.axvline(x=tau, color="red", linestyle="--", label=f"Max τ={tau}")
        ax.axhline(y=0, color="gray", linewidth=0.7)
        ax.set_title(f"{label} - Y")
        ax.set_ylabel("r(τ)")
        ax.set_xlabel("Сдвиг τ")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"crosscorrelations_{label} → Y.png")
        plt.clf()

    # Строим обучающую выборку для ИНС-модели

    # Определяем значение сдвига, которое будет использовано для формирования обучающей выборки
    max_lambda = max(tau1, tau2, tau3)
    print(f"Максимальный сдвиг: {max_lambda}")

    # Общее число членов временного ряда
    k = len(y)

    # Определяем дальность прогноза
    h = 20

    # Определяем максимальное число входов нейронной сети
    n = int((-5 + np.sqrt(25 + 16 * (k - h))) / 8)
    print(f"Максимальное число входов: {n}")

    # Определяем количество точек предыстории
    l = 3
    n_input = 3 * l
    print(f"Вывод: определяем {n_input} точек на {l} входов")

    X, Y = build_dataset_for_nlp(x1, x2, x3, y, tau1, tau2, tau3, max_lambda, l)
    print(f"Размер выборки: X={X.shape}, Y={y.shape}")

    # Проверяем условие P1 <= P2 (т. е. что выбранное l подходит)
    P2 = len(Y)
    P1 = 4 * n_input**2 + 4 * n_input + 1
    print(f"Проверка условия: P1 = {P1}, P2 = {P2}. P1 <= P2: {P1 <= P2}")

    # Проводим обучение ИНС

        # Нормируем данные
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()

    # Делим на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False
    )

    # Реальные значения теста — обратное преобразование из нормированных
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    print(f"Обучающая выборка: {len(X_train)} примеров")
    print(f"Тестовая выборка: {len(X_test)} примеров")

    # Сеть (по теореме Колмогорова): входной слой -> скрытый слой -> выход
    hidden_sz = n_input * (2 * n_input + 1)
    print(f"Структура сети: {n_input} - {hidden_sz} - 1")

    model = Sequential([
        Dense(hidden_sz, activation="relu", input_shape=(n_input,)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        verbose=0
    )

    # Прогноз на тесте
    pred_scaled = model.predict(X_test, verbose=0).ravel()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    # Оценка качества на тесте
    mse = mean_squared_error(y_test_real, pred)
    r2 = r2_score(y_test_real, pred)
    rel_err = np.abs((y_test_real - pred) / y_test_real) * 100

    print("Результаты на тесте:")
    print(f"MSE = {mse:.4f}")
    print(f"R2 = {r2:.4f}")
    print(f"Средняя ошибка = {np.mean(rel_err):.2f}%")
    print(f"Максимальная ошибка = {np.max(rel_err):.2f}%")

    # Прогноз на 15 шагов вперед
    print("Прогноз на 15 шагов:")
    predict = []
    for step in range(15):
        t = len(Y) + step
        row = []

        for d in range(-l + 1, 1):
            row.append(get_value(x1, t - max_lambda + tau1 + d))

        for d in range(-l + 1, 1):
            row.append(get_value(x2, t - max_lambda + tau2 + d))

        for d in range(-l + 1, 1):
            row.append(get_value(x3, t - max_lambda + tau3 + d))

        row_scaled = scaler_x.transform(np.array(row).reshape(1, -1))
        pred_scaled_step = model.predict(row_scaled, verbose=0).ravel()
        pred_val = scaler_y.inverse_transform(pred_scaled_step.reshape(-1, 1)).ravel()[0]

        predict.append(pred_val)

    actual_last15 = Y[-15:]
    for i in range(15):
        err = abs(predict[i] - actual_last15[i])
        rel = err / actual_last15[i] * 100 if actual_last15[i] != 0 else 0
        print(f"  {i + 1}: прогноз={predict[i]:.2f}, факт={actual_last15[i]:.2f}, ошибка={rel:.1f}%")

    print(
        f"Средняя ошибка прогноза: {np.mean(np.abs((actual_last15 - predict) / actual_last15)) * 100:.2f}%"
    )

    # Рисуем графики
    train_sz = len(X_train)

    # График 1: Тестовая выборка
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(train_sz, len(X_scaled)), y_test_real, color="orange", marker="s", linestyle="-", label="Реальные", markersize=3)
    ax.plot(range(train_sz, len(X_scaled)), pred, color="purple", marker="s", linestyle="-", label="Предсказанные", markersize=3)
    ax.set_title(f"Тест (R2={r2:.3f})")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig("plot1_test.png", dpi=150)
    plt.close()

    # График 2: Диаграмма рассеяния
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test_real, pred, alpha=0.6)
    ax.plot(
        [y_test_real.min(), y_test_real.max()],
        [y_test_real.min(), y_test_real.max()],
        "r--",
    )
    ax.set_title("Реальные значения vs Предсказанные значения")
    ax.set_xlabel("Реальные")
    ax.set_ylabel("Предсказанные")
    ax.grid()
    plt.tight_layout()
    plt.savefig("plot2_scatter.png", dpi=150)
    plt.close()

    # График 3: Прогноз на 15 шагов
    x_f = np.arange(len(Y) - 15, len(Y))
    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(range(len(Y)), Y, "b-", alpha=0.5, label="История")
    ax.plot(x_f, actual_last15, "bo-", label="Факт")
    ax.plot(x_f, predict, "rs--", label="Предсказанные")
    ax.axvline(x=len(Y) - 15, color="r", linestyle="--", label="Начало прогноза")
    ax.set_title("Предсказание на 15 шагов")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig("plot3_predict.png", dpi=150)
    plt.close()

    # График 4: Кривая потерь при обучении
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"], color="blue")
    ax.set_title("Кривая обучения (MSE по эпохам)")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.grid()
    plt.tight_layout()
    plt.savefig("plot4_loss_curve.png", dpi=150)
    plt.close()

    # График 5: Относительная погрешность на тестовой выборке
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rel_err, color="darkorange", marker="o", markersize=3)
    ax.axhline(y=np.mean(rel_err), color="r", linestyle="--", label=f"Среднее={np.mean(rel_err):.2f}%")
    ax.set_title("Относительная погрешность на тесте")
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Ошибка, %")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig("plot5_rel_error.png", dpi=150)
    plt.close()