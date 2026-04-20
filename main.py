import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("Прогнозирование множественного временного ряда")
print("=" * 70)

# Загружаем данные из Excel
file_path = 'Данные_к_лаб_4.xlsx'
df = pd.read_excel(file_path, sheet_name='Лист1')
data = df.iloc[:310].copy()

x1 = data['x1'].values
x2 = data['x2'].values
x3 = data['x3'].values
Y = data['Y'].values

print(f"\nЗагружено {len(Y)} наблюдений")
print(f"Y от {Y.min():.2f} до {Y.max():.2f}")


# Функция для расчета взаимной корреляции
def calc_corr(x, y, max_lag=30):
    n = len(x)
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            x_tr = x[-lag:]
            y_tr = y[:n + lag]
        elif lag > 0:
            x_tr = x[:n - lag]
            y_tr = y[lag:]
        else:
            x_tr = x
            y_tr = y
        if len(x_tr) > 2:
            r = np.corrcoef(x_tr, y_tr)[0, 1]
            corrs.append(r if not np.isnan(r) else 0)
        else:
            corrs.append(0)
    return np.array(lags), np.array(corrs)


# Считаем корреляции для каждого входа
lags, corr1 = calc_corr(x1, Y)
_, corr2 = calc_corr(x2, Y)
_, corr3 = calc_corr(x3, Y)

# Находим сдвиги
tau1 = lags[np.argmax(np.abs(corr1))]
tau2 = lags[np.argmax(np.abs(corr2))]
tau3 = lags[np.argmax(np.abs(corr3))]

print("\nРезультаты корреляционного анализа:")
print(f"  x1->Y: сдвиг={tau1}, макс.корр={np.max(np.abs(corr1)):.4f}")
print(f"  x2->Y: сдвиг={tau2}, макс.корр={np.max(np.abs(corr2)):.4f}")
print(f"  x3->Y: сдвиг={tau3}, макс.корр={np.max(np.abs(corr3)):.4f}")

lambda_shift = max(tau1, tau2, tau3, 0)
print(f"  Максимальный сдвиг λ={lambda_shift}")

# Проверяем сколько входов можно использовать
k_total = len(Y)
h = 10
n_max = int((-5 + np.sqrt(25 + 16 * (k_total - h))) / 8)
print(f"\nМаксимальное число входов по формуле (11): n_max={n_max}")

# Берем l=2 (по 2 точки на каждый вход) -> всего 6 входов
l_hist = 2
n_inputs = 3 * l_hist
print(f"Выбрано l={l_hist}, всего входов n={n_inputs}")


# Формируем обучающую выборку
def build_data(x1, x2, x3, Y, tau1, tau2, tau3, lam, l):
    X_list = []
    Y_list = []
    min_idx = lam + max(0, -tau1, -tau2, -tau3) + l
    max_idx = len(Y) - 1

    for t in range(min_idx, max_idx):
        row = []

        # x1
        for d in range(-l + 1, 1):
            idx = t - lam + tau1 + d
            if 0 <= idx < len(x1):
                row.append(x1[idx])
            else:
                row.append(x1[0] if idx < 0 else x1[-1])

        # x2
        for d in range(-l + 1, 1):
            idx = t - lam + tau2 + d
            if 0 <= idx < len(x2):
                row.append(x2[idx])
            else:
                row.append(x2[0] if idx < 0 else x2[-1])

        # x3
        for d in range(-l + 1, 1):
            idx = t - lam + tau3 + d
            if 0 <= idx < len(x3):
                row.append(x3[idx])
            else:
                row.append(x3[0] if idx < 0 else x3[-1])

        X_list.append(row)
        Y_list.append(Y[t])

    return np.array(X_list), np.array(Y_list)


X, y = build_data(x1, x2, x3, Y, tau1, tau2, tau3, lambda_shift, l_hist)
print(f"\nРазмер выборки: X={X.shape}, y={y.shape}")

# Проверяем условие P1 <= P2
P2 = len(y)
P1 = 4 * n_inputs ** 2 + 4 * n_inputs + 1
print(f"\nПроверка условия (9): P1={P1}, P2={P2}")
print(f"P1 <= P2: {P1 <= P2}")

# Нормируем данные
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Делим на обучение и тест (80/20, сохраняя порядок)
train_sz = int(0.8 * len(X_scaled))
X_train = X_scaled[:train_sz]
y_train = y_scaled[:train_sz]
X_test = X_scaled[train_sz:]
y_test = y_scaled[train_sz:]
y_test_real = y[train_sz:]

print(f"Обучающая выборка: {len(X_train)} примеров")
print(f"Тестовая выборка: {len(X_test)} примеров")

# Создаем сеть: входной слой -> скрытый (по теореме Колмогорова) -> выход
hidden_sz = n_inputs * (2 * n_inputs + 1)
print(f"\nСтруктура сети: {n_inputs} -> {hidden_sz} -> 1")

mlp = MLPRegressor(
    hidden_layer_sizes=(hidden_sz,),
    activation='tanh',
    max_iter=1000,
    random_state=42
)

print("Обучение сети...")
mlp.fit(X_train, y_train)

# Прогноз на тесте
pred_scaled = mlp.predict(X_test)
pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

# Оценка качества
mse = mean_squared_error(y_test_real, pred)
r2 = r2_score(y_test_real, pred)
rel_err = np.abs((y_test_real - pred) / y_test_real) * 100

print("\nРезультаты на тесте:")
print(f"  MSE={mse:.4f}")
print(f"  R2={r2:.4f}")
print(f"  Средняя ошибка={np.mean(rel_err):.2f}%")
print(f"  Макс ошибка={np.max(rel_err):.2f}%")

# Прогноз на 10 шагов вперед
print("\nПрогноз на 10 шагов:")
forecast = []
for step in range(10):
    t = len(Y) + step
    row = []

    for d in range(-l_hist + 1, 1):
        idx = t - lambda_shift + tau1 + d
        if idx < len(x1):
            row.append(x1[idx] if idx >= 0 else x1[0])
        else:
            row.append(x1[-1])

    for d in range(-l_hist + 1, 1):
        idx = t - lambda_shift + tau2 + d
        if idx < len(x2):
            row.append(x2[idx] if idx >= 0 else x2[0])
        else:
            row.append(x2[-1])

    for d in range(-l_hist + 1, 1):
        idx = t - lambda_shift + tau3 + d
        if idx < len(x3):
            row.append(x3[idx] if idx >= 0 else x3[0])
        else:
            row.append(x3[-1])

    row_scaled = scaler_x.transform(np.array(row).reshape(1, -1))
    pred_scaled = mlp.predict(row_scaled)
    pred_val = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
    forecast.append(pred_val)

actual_last10 = Y[-10:]
for i in range(10):
    err = abs(forecast[i] - actual_last10[i])
    rel = err / actual_last10[i] * 100 if actual_last10[i] != 0 else 0
    print(f"  {i + 1}: прогноз={forecast[i]:.2f}, факт={actual_last10[i]:.2f}, ошибка={rel:.1f}%")

print(f"\nСредняя ошибка прогноза: {np.mean(np.abs((actual_last10 - forecast) / actual_last10)) * 100:.2f}%")

# Рисуем графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Тестовая выборка
axes[0, 0].plot(range(train_sz, len(X_scaled)), y_test_real, 'b-o', label='Факт', markersize=3)
axes[0, 0].plot(range(train_sz, len(X_scaled)), pred, 'r-s', label='Прогноз', markersize=3)
axes[0, 0].set_title(f'Тест (R²={r2:.3f})')
axes[0, 0].legend()
axes[0, 0].grid()

# Диаграмма рассеяния
axes[0, 1].scatter(y_test_real, pred, alpha=0.6)
axes[0, 1].plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--')
axes[0, 1].set_title('Факт vs Прогноз')
axes[0, 1].grid()

# Прогноз на 10 шагов
x_f = np.arange(len(Y) - 10, len(Y))
axes[1, 0].plot(range(len(Y)), Y, 'b-', alpha=0.5, label='История')
axes[1, 0].plot(x_f, actual_last10, 'bo-', label='Факт')
axes[1, 0].plot(x_f, forecast, 'rs--', label='Прогноз')
axes[1, 0].axvline(x=len(Y) - 10, color='r', linestyle='--', label='Начало прогноза')
axes[1, 0].set_title('Прогноз на 10 шагов')
axes[1, 0].legend()
axes[1, 0].grid()

# Остатки
res = y_test_real - pred
axes[1, 1].bar(range(len(res)), res, color='coral', alpha=0.7)
axes[1, 1].axhline(y=0, color='k')
axes[1, 1].axhline(y=np.mean(res), color='r', linestyle='--', label=f'Среднее={np.mean(res):.3f}')
axes[1, 1].set_title('Остатки')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.savefig('part2_results.png', dpi=150)
plt.show()