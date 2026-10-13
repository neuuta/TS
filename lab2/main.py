import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. ГЕНЕРАЦІЯ ДАНИХ (Варіант 1: Квадратичний тренд + Рівномірний шум)
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10, n_samples)

# Квадратичний ідеальний тренд
ideal_trend = 2 * t**2 - 5 * t + 10

# Додавання рівномірного шуму
noise = np.random.uniform(-15, 15, n_samples)
data_noisy = ideal_trend + noise

# Додавання аномальних вимірів (Група вимог 2)
anomaly_fraction = 0.05
num_anomalies = int(n_samples * anomaly_fraction)
anomaly_indices = np.random.choice(n_samples, num_anomalies, replace=False)

data_with_anomalies = data_noisy.copy()
for idx in anomaly_indices:
    data_with_anomalies[idx] += np.random.choice([-1, 1]) * np.random.uniform(60, 100)

# 2. R&D АЛГОРИТМ ОЧИЩЕННЯ (Група вимог 3)
def custom_adaptive_anomaly_filter(y, window_size=30, threshold=2.0):
    """
    R&D Алгоритм: Адаптивний локальний медіанний фільтр для нестаціонарних трендів.
    """
    y_clean = y.copy()
    anomalies_detected = []
    
    for i in range(len(y)):
        start = max(0, i - window_size // 2)
        end = min(len(y), i + window_size // 2)
        window = y[start:end]
        
        local_median = np.median(window)
        local_std = np.std(window)
        
        # Перевірка на аномальність відносно локального вікна
        if local_std > 0 and abs(y[i] - local_median) > threshold * local_std:
            y_clean[i] = local_median
            anomalies_detected.append(i)
            
    print(f"[R&D Filter] Виявлено та очищено аномалій: {len(anomalies_detected)}")
    return y_clean

data_cleaned = custom_adaptive_anomaly_filter(data_with_anomalies)

# 3. СТАТИСТИЧНЕ НАВЧАННЯ МНК ТА ОПТИМІЗАЦІЯ
# Обмежено ступені для уникнення перенавчання
degrees_to_test = [1, 2, 3] 
best_degree = 1
best_r2 = -float('inf')
best_poly_model = None

print("\n--- Оптимізація поліноміальної моделі ---")
for deg in degrees_to_test:
    coeffs = np.polyfit(t, data_cleaned, deg)
    poly = np.poly1d(coeffs)
    y_pred = poly(t)
    
    mse = mean_squared_error(data_cleaned, y_pred)
    r2 = r2_score(data_cleaned, y_pred)
    print(f"Ступінь {deg}: MSE = {mse:.2f}, R^2 = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_degree = deg
        best_poly_model = poly

print(f"\n=> Оптимальна модель обрана зі ступенем {best_degree} (R^2 = {best_r2:.4f})")

# 4. ЕКСТРАПОЛЯЦІЯ / ПРОГНОЗУВАННЯ
n_extrapolate = int(n_samples * 0.5)
t_extrapolate = np.linspace(10, 15, n_extrapolate)
y_forecast = best_poly_model(t_extrapolate)

# 5. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
plt.figure(figsize=(14, 10))

# Графік 1: Вхідні дані з аномаліями
plt.subplot(2, 1, 1)
plt.scatter(t, data_with_anomalies, color='red', alpha=0.5, s=10, label='Дані з аномаліями')
plt.plot(t, ideal_trend, color='black', linewidth=2, label='Ідеальний квадратичний тренд')
plt.title('Вхідний Big Data масив з аномальними вимірами (Варіант 1)')
plt.legend()
plt.grid(True)

# Графік 2: Очищені дані, навчена модель та прогноз
plt.subplot(2, 1, 2)
plt.scatter(t, data_cleaned, color='lightblue', alpha=0.6, s=10, label='Очищені дані (R&D фільтр)')
plt.plot(t, best_poly_model(t), color='blue', linewidth=2.5, label=f'Навчена МНК модель (ступінь {best_degree})')
plt.plot(t_extrapolate, y_forecast, color='green', linewidth=2.5, linestyle='--', label='Екстраполяція (прогноз на 0.5 інтервалу)')

plt.axvline(x=10, color='grey', linestyle=':', linewidth=2)
plt.text(10.2, min(data_cleaned), 'Зона прогнозування ->', color='green', fontsize=12)

plt.title('Результати статистичного навчання та прогнозування')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()