import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def generate_data(n_samples, change=False):
    t = np.arange(n_samples)
    ideal_trend = 10 + 0.01 * t + 5e-5 * t**2
    if change:
        dt = np.clip(t - int(0.6 * n_samples), 0, None)
        ideal_trend = ideal_trend - 0.35 * dt - 2e-4 * dt**2

    noise = np.random.uniform(-15, 15, n_samples)
    data = ideal_trend + noise

    num_anomalies = int(n_samples * 0.05)
    anomaly_indices = np.random.choice(n_samples, num_anomalies, replace=False)
    for idx in anomaly_indices:
        data[idx] += np.random.choice([-1, 1]) * np.random.uniform(40, 70)

    return t, ideal_trend, data


def clean_anomalies(y, window_size=30, threshold=3.0):
    y_clean = y.copy()
    anomalies = 0
    for i in range(len(y)):
        start = max(0, i - window_size)
        window = y[start:i + 1]
        if len(window) < 10:
            continue
        local_median = np.median(window)
        local_std = 1.4826 * np.median(np.abs(window - local_median))
        if abs(y[i] - local_median) > threshold * local_std:
            y_clean[i] = local_median
            anomalies += 1
    print(f"Виявлено та очищено аномалій: {anomalies}")
    return y_clean


def abf(y):
    x, v = 0.0, 0.0
    out = np.zeros(len(y))
    for k in range(len(y)):
        x_p = x + v
        r = y[k] - x_p
        alpha = 2 * (2 * k + 1) / ((k + 1) * (k + 2))
        beta = 6 / ((k + 1) * (k + 2))
        x = x_p + alpha * r
        v = v + beta * r
        out[k] = x
    return out


def abg_gains(m):
    d = (m + 1) * (m + 2) * (m + 3)
    return 3 * (3 * m * m + 3 * m + 2) / d, 18 * (2 * m + 1) / d, 60 / d


def abgf(y):
    x, v, a = 0.0, 0.0, 0.0
    out = np.zeros(len(y))
    for k in range(len(y)):
        x_p = x + v + 0.5 * a
        v_p = v + a
        r = y[k] - x_p
        alpha, beta, gamma = abg_gains(k)
        x = x_p + alpha * r
        v = v_p + beta * r
        a = a + gamma * r
        out[k] = x
    return out


def adaptive_abgf(y, window=20, z_crit=3.29, m_min=20):
    d3 = y[3:] - 3 * y[2:-1] + 3 * y[1:-2] - y[:-3]
    sigma = 1.4826 * np.median(np.abs(d3 - np.median(d3))) / np.sqrt(20)
    chi_crit = window + z_crit * np.sqrt(2 * window)

    x, v, a = 0.0, 0.0, 0.0
    m = 0
    out = np.zeros(len(y))
    memory = np.zeros(len(y))
    z_buf = []
    alarms = 0

    for k in range(len(y)):
        x_p = x + v + 0.5 * a
        v_p = v + a
        r = y[k] - x_p
        alpha, beta, gamma = abg_gains(m)

        if m > 3:
            z_buf.append(np.clip(r / (sigma * np.sqrt(1 + alpha)), -3, 3))
        if len(z_buf) > window:
            z_buf.pop(0)

        if len(z_buf) == window and m > m_min:
            bias = abs(np.sum(z_buf)) / np.sqrt(window)
            var = np.sum(np.square(z_buf))
            if bias > z_crit or var > chi_crit:
                m = max(m_min, m // 2)
                alpha, beta, gamma = abg_gains(m)
                z_buf = []
                alarms += 1

        x = x_p + alpha * r
        v = v_p + beta * r
        a = a + gamma * r
        out[k] = x
        memory[k] = m
        m += 1

    print(f"Адаптивний фільтр: виявлено розходжень = {alarms}")
    return out, memory


def print_metrics(name, trend, est, skip=50):
    e = est[skip:] - trend[skip:]
    mse = mean_squared_error(trend[skip:], est[skip:])
    r2 = r2_score(trend[skip:], est[skip:])
    print(f"{name:<22} зміщення = {np.mean(e):8.3f}   MSE = {mse:9.3f}   R^2 = {r2:.4f}   |e| в кінці = {abs(e[-1]):.3f}")


def run_scenario(title, change):
    print(f"\n========== {title} ==========")
    t, trend, data = generate_data(2000, change)
    data_clean = clean_anomalies(data)

    y_abf = abf(data_clean)
    y_abgf = abgf(data_clean)
    y_adapt, memory = adaptive_abgf(data_clean)

    print_metrics("alfa-beta", trend, y_abf)
    print_metrics("alfa-beta-gamma", trend, y_abgf)
    print_metrics("адаптивний a-b-g", trend, y_adapt)

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.scatter(t, data, color='red', alpha=0.4, s=5, label='Дані з аномаліями')
    plt.scatter(t, data_clean, color='lightblue', alpha=0.6, s=5, label='Очищені дані')
    plt.plot(t, trend, color='black', linewidth=2, label='Ідеальний тренд')
    plt.plot(t, y_abf, color='orange', linewidth=2, label='alfa-beta')
    plt.plot(t, y_abgf, color='green', linewidth=2, label='alfa-beta-gamma')
    plt.plot(t, y_adapt, color='blue', linewidth=2, label='Адаптивний alfa-beta-gamma')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, y_abf - trend, color='orange', label='alfa-beta')
    plt.plot(t, y_abgf - trend, color='green', label='alfa-beta-gamma')
    plt.plot(t, y_adapt - trend, color='blue', label='Адаптивний alfa-beta-gamma')
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-40, 40)
    plt.title('Похибка оцінки відносно ідеального тренду')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, memory, color='blue')
    plt.yscale('log')
    plt.title("Пам'ять адаптивного фільтра m(k)")
    plt.grid(True)

    plt.tight_layout()


np.random.seed(42)
run_scenario('Варіант 1: квадратичний тренд + рівномірний шум', change=False)
run_scenario('Зміна динаміки тренду (перевірка ефекту розходження)', change=True)
plt.show()
