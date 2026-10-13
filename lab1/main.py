import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
import requests

# Функція генерації нормального шуму
def random_norm(dm, dsig, iter):
    S = np.random.normal(dm, dsig, iter)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('--- статистичні характеристики НОРМАЛЬНОГО закону розподілу ---')
    print('Математичне сподівання =', mS)
    print('Дисперсія =', dS)
    print('СКВ =', scvS)
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title("Гістограма нормального розподілу")
    plt.show()
    return S

# Функція генерації рівномірного шуму
def random_uniform(low, high, iter):
    S = np.random.uniform(low, high, iter)
    mS = np.mean(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('--- статистичні характеристики РІВНОМІРНОГО закону розподілу ---')
    print('Математичне сподівання =', mS)
    print('Дисперсія =', dS)
    print('СКВ =', scvS)
    plt.hist(S, bins=20, facecolor="green", alpha=0.5)
    plt.title("Гістограма рівномірного розподілу")
    plt.show()
    return S

# Модель постійного тренду
def Model_Constant(n, constant_value=10):
    S0 = np.full(n, constant_value, dtype=float)
    return S0

# Модель квадратичного тренду
def Model_Quadratic(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.000005 * (i ** 2)
    return S0

# Адитивна модель: тренд + шум
def Model_Additive(Trend, Noise, n):
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = Trend[i] + Noise[i]
    return SV

# Адитивна модель з аномальними вимірами
def Model_Additive_AV(Trend, Noise, n, NAV, Q_AV, dsig):
    SV_AV = Model_Additive(Trend, Noise, n)
    anomaly_indices = np.random.randint(0, n, NAV)
    for idx in anomaly_indices:
        SV_AV[idx] += Q_AV * dsig * np.random.choice([-1, 1])
    return SV_AV

# Статистичні характеристики вибірки
def Stat_characteristics(SL, Text):
    mS = np.mean(SL)
    dS = np.var(SL)
    scvS = mt.sqrt(dS)
    print('---', Text, '---')
    print('Математичне сподівання =', mS)
    print('Дисперсія =', dS)
    print('СКВ =', scvS)
    print('-'*40)

# Графіки
def Plot_AV(Trend, SV_L, Text):
    plt.clf()
    plt.plot(SV_L, label="Вибірка", color="lightblue")
    plt.plot(Trend, label="Ідеальний тренд", color="orange")
    plt.title(Text)
    plt.legend()
    plt.show()

# ФУНКЦІЯ парсингу реальних даних
def file_parsing(File_name, Data_names):
    try:
        d = pd.read_excel(File_name)
        results = []
        for name in Data_names:
            values = d[name].dropna().values
            results.append(values)
            print(f"--- Аналіз показника: {name} ---")
            Stat_characteristics(values, f"Реальні дані: {name}")
        return results
    except Exception as e:
        print("Помилка читання файлу реальних даних:", e)
        return []

def parse_custom_site_and_save():
    url = 'https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json'
    print(f"Починаємо парсинг сайту: {url}")
    
    response = requests.get(url)
    parsed_data = []
    if response.status_code == 200:
        data = response.json() 
        for item in data[:20]:
            parsed_data.append(item['rate'])
    else:
        print(f"Помилка з'єднання: {response.status_code}")
            
    df = pd.DataFrame({'Parsed_Rate': parsed_data})
    df.to_csv('parsed_custom_data.csv', index=False)
    print("Дані успішно збережено у файл 'parsed_custom_data.csv'")
    return np.array(parsed_data)

def level_3_synthesis(real_data):
    if len(real_data) == 0: return
    mS_real = np.mean(real_data)
    dS_real = np.var(real_data)
    scvS_real = mt.sqrt(dS_real)
    
    print('--- Статистичні характеристики результатів парсингу ---')
    print('Математичне сподівання =', mS_real)
    print('Дисперсія =', dS_real)
    print('СКВ =', scvS_real)

    n = len(real_data)
    synthesized_trend = np.full(n, mS_real)
    synthesized_noise = np.random.normal(0, scvS_real, n)
    synthesized_data = synthesized_trend + synthesized_noise

    plt.clf()
    plt.plot(real_data, label="Реальні дані (Парсинг)", marker='o')
    plt.plot(synthesized_data, label="Синтезована модель", linestyle='--')
    plt.title("Порівняння реальних (з парсингу) та синтезованих даних")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 10000
    iter = int(n)
    Q_AV = 5
    NAV = int(iter * 0.01)
    
    dm = 0
    dsig = 5
    
    # Моделі ідеального тренду
    Trend_Constant = Model_Constant(n, 20)
    Trend_Quadratic = Model_Quadratic(n)
    
    # Моделі шуму
    Noise_Norm = random_norm(dm, dsig, iter)
    Noise_Unif = random_uniform(-8, 8, iter)
    
    # Комбінація 1: Квадратична модель + Нормальний шум
    SV_QN = Model_Additive(Trend_Quadratic, Noise_Norm, n)
    Plot_AV(Trend_Quadratic, SV_QN, 'Квадратична модель + Нормальний шум')
    Stat_characteristics(SV_QN, 'Вибірка (Квадратична + Нормальний)')
    
    # Комбінація 2: Постійна модель + Рівномірний шум
    SV_CU = Model_Additive(Trend_Constant, Noise_Unif, n)
    Plot_AV(Trend_Constant, SV_CU, 'Постійна модель + Рівномірний шум')
    Stat_characteristics(SV_CU, 'Вибірка (Постійна + Рівномірний)')
    
    # Комбінація 3: Квадратична модель + Нормальний шум + Аномалії
    SV_QN_AV = Model_Additive_AV(Trend_Quadratic, Noise_Norm, n, NAV, Q_AV, dsig)
    Plot_AV(Trend_Quadratic, SV_QN_AV, 'Квадратична модель + Норм. шум + Аномальні виміри')
    Stat_characteristics(SV_QN_AV, 'Вибірка з Аномаліями (Квадратична + Норм)')
    
    # Обробка реальних даних
    file_parsing('Oschadbank (USD).xls', ['Продаж', 'Купівля', 'Курс НБУ'])
    
    print("\n--- ВИКОНАННЯ 3 РІВНЯ СКЛАДНОСТІ ---")
    parsed_array = parse_custom_site_and_save()
    level_3_synthesis(parsed_array)