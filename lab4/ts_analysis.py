import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_real_data(filename):
    d = pd.read_excel(filename)
    d.index = pd.to_datetime(d['Дата'], dayfirst=True)
    d = d[['Купівля', 'Продаж', 'КурсНбу']]
    print(f"Пропущені (нульові) значення: {(d == 0).sum().to_dict()} -> замінено попереднім значенням")
    return d.replace(0, np.nan).ffill().bfill()


def hurst(x):
    x = np.asarray(x, dtype=float)
    sizes = [8, 16, 32, 64, 128]
    rs = []
    for n in sizes:
        values = []
        for i in range(0, len(x) - n + 1, n):
            seg = np.diff(x[i:i + n])
            z = np.cumsum(seg - seg.mean())
            s = seg.std()
            if s > 0:
                values.append((z.max() - z.min()) / s)
        rs.append(np.mean(values))
    return np.polyfit(np.log(sizes), np.log(rs), 1)[0]


def properties(x):
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    return {
        'середнє': np.mean(x),
        'СКВ': np.std(x),
        'сер. приріст': np.mean(d),
        'СКВ приросту': np.std(d),
        'частка нульових змін': np.mean(d == 0),
        'ексцес приросту': stats.kurtosis(d),
        'ADF p (ряд)': adfuller(x)[1],
        'ADF p (приріст)': adfuller(d)[1],
        'показник Херста': hurst(x),
    }


def print_properties(title, props):
    print(f"\n--- Властивості: {title} ---")
    for k, v in props.items():
        print(f"{k:<24} {v:10.4f}")


def decomposition(ts, period=20):
    res_add = seasonal_decompose(ts, model='additive', period=period)
    res_mul = seasonal_decompose(ts, model='multiplicative', period=period)
    resid = res_add.resid.dropna()
    print("\n--- Декомпозиція (адитивна модель, період 20 днів) ---")
    print(f"розмах тренду = {res_add.trend.max() - res_add.trend.min():.3f} грн")
    print(f"амплітуда сезонної складової = {res_add.seasonal.max() - res_add.seasonal.min():.3f} грн")
    print(f"СКВ залишків = {resid.std():.3f} грн")
    print(f"мультиплікативна сезонність: {res_mul.seasonal.min():.4f} ... {res_mul.seasonal.max():.4f}")

    fig = res_add.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle('Адитивна декомпозиція курсу купівлі USD')
    fig.tight_layout()
    return res_add


def correlation(df):
    changes = df.diff().dropna()
    print("\n--- Кореляційна матриця рівнів ---")
    print(df.corr().round(4))
    print("\n--- Кореляційна матриця щоденних змін ---")
    print(changes.corr().round(4))

    a = changes['КурсНбу'].values
    b = changes['Купівля'].values
    lags = range(-5, 6)
    cc = [np.corrcoef(a[max(0, -l):len(a) - max(0, l)], b[max(0, l):len(b) - max(0, -l)])[0, 1] for l in lags]
    best = list(lags)[int(np.argmax(cc))]
    print("\nВзаємна кореляція змін (НБУ -> Купівля) за лагами:")
    print(' '.join(f"{l:+d}:{c:.2f}" for l, c in zip(lags, cc)))
    print(f"максимум кореляції на лагу {best:+d} дн.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(changes.corr(), cmap='gray_r', vmin=0, vmax=1)
    plt.xticks(range(3), changes.columns)
    plt.yticks(range(3), changes.columns)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{changes.corr().iloc[i, j]:.2f}", ha='center', va='center', color='red')
    plt.title('Кореляція щоденних змін')
    plt.subplot(1, 2, 2)
    plt.bar(list(lags), cc, color='steelblue')
    plt.title('Взаємна кореляція змін НБУ та курсу купівлі')
    plt.xlabel('лаг, днів')
    plt.grid(True)
    plt.tight_layout()


def clustering(ts, window=10, n_clusters=3):
    x = ts.values
    d = np.diff(x)
    n_win = len(d) // window
    feats = np.array([[d[i * window:(i + 1) * window].mean(), d[i * window:(i + 1) * window].std()]
                      for i in range(n_win)])
    X = StandardScaler().fit_transform(np.column_stack([feats[:, 0], np.log(feats[:, 1] + 0.01)]))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
    order = np.argsort([feats[km.labels_ == c, 1].mean() for c in range(n_clusters)])
    names = ['стабільний', 'помірний', 'кризовий']
    labels = np.array([names[int(np.where(order == c)[0][0])] for c in km.labels_])

    print("\n--- Кластеризація 10-денних сегментів (KMeans) ---")
    for name in names:
        m = labels == name
        print(f"{name:<11} сегментів = {m.sum():3d}   сер. зміна = {feats[m, 0].mean():7.4f}   СКВ змін = {feats[m, 1].mean():.4f}")

    plt.figure(figsize=(12, 4))
    colors = {'стабільний': 'green', 'помірний': 'orange', 'кризовий': 'red'}
    for i in range(n_win):
        seg = ts.iloc[i * window:(i + 1) * window + 1]
        plt.plot(seg.index, seg.values, color=colors[labels[i]], linewidth=2)
    for name, c in colors.items():
        plt.plot([], [], color=c, label=name)
    plt.title('Кластери режимів курсу купівлі USD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return labels


def plot_real(df):
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title('Курс USD, Ощадбанк')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.hist(np.diff(df['Купівля'].values), bins=50, color='steelblue')
    plt.title('Гістограма щоденних змін курсу купівлі')
    plt.grid(True)
    plt.tight_layout()
