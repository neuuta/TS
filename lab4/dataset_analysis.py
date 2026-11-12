import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

AGGREGATES = ['Sales', 'VNI', 'Other', 'REGION', 'Region', 'States', 'Countries', 'Brokers', 'Subsidiaries',
              'Importers', 'Africa', 'Ocean', 'Territories', 'Caucas', 'CEIS', 'G9', 'Nordic', 'Iberia', 'Benelux',
              'Mid CE', 'Adriatic', 'Central', 'WORLDWIDE', 'INTERNATIONAL', 'International', 'Total', 'CKD', 'CBU',
              'KSA', 'GCC', 'RME', 'DT ', 'ISR', 'DOM', 'ASEAN', 'IRAN', 'MOROCCO', 'Korea', 'Poland+',
              'Central Asia', 'Europe']
EXACT = ['South', 'North', 'East', 'ACC', 'Others India']


def load_dataset(filename):
    d = pd.read_excel(filename, header=None)
    d = d.iloc[9:].copy()
    df = pd.DataFrame({
        'країна': d[0].astype(str).str.strip(),
        'TIV': pd.to_numeric(d[14], errors='coerce'),
        'TIV_y1': pd.to_numeric(d[15], errors='coerce'),
        'обсяг': pd.to_numeric(d[18], errors='coerce'),
        'обсяг_y1': pd.to_numeric(d[19], errors='coerce'),
        'MS': pd.to_numeric(d[22], errors='coerce'),
        'MS_y1': pd.to_numeric(d[23], errors='coerce'),
        'MS_грудень': pd.to_numeric(d[10], errors='coerce'),
    })
    df = df.dropna()
    mask = df['країна'].apply(lambda s: not any(a in s for a in AGGREGATES) and s not in EXACT)
    df = df[mask & (df['TIV'] >= 5) & (df['MS'] < 100)].drop_duplicates('країна')
    df['ринок_ріст_%'] = (df['TIV'] / df['TIV_y1'] - 1) * 100
    df['logTIV'] = np.log10(df['TIV'] * 1000)
    print(f"Data_Set_1 (Renault World Sales 2020): відібрано країн = {len(df)}")
    return df.reset_index(drop=True)


def describe(df):
    print("\n--- Властивості показників Data_Set_1 ---")
    for col in ['TIV', 'обсяг', 'MS', 'ринок_ріст_%']:
        v = df[col]
        print(f"{col:<14} сер. = {v.mean():10.2f}  медіана = {v.median():9.2f}  СКВ = {v.std():10.2f}  "
              f"асиметрія = {stats.skew(v):6.2f}  ексцес = {stats.kurtosis(v):7.2f}")
    top = df.sort_values('обсяг', ascending=False).head(10)
    print("\nТоп-10 країн за продажами Renault у 2020 р.:")
    print(top[['країна', 'TIV', 'обсяг', 'MS']].to_string(index=False))


def decomposition(df):
    dv = df['обсяг'] - df['обсяг_y1']
    market = (df['TIV'] - df['TIV_y1']) * 1000 * df['MS_y1'] / 100
    share = df['TIV'] * 1000 * (df['MS'] - df['MS_y1']) / 100
    print("\n--- Декомпозиція зміни продажів: обсяг = ринок × частка ---")
    print(f"сумарна зміна продажів = {dv.sum():10.0f} од.")
    print(f"ефект ринку            = {market.sum():10.0f} од.")
    print(f"ефект частки ринку     = {share.sum():10.0f} од.")
    t = df.assign(ринок=market, частка=share).sort_values('обсяг', ascending=False).head(10)

    plt.figure(figsize=(12, 5))
    x = np.arange(len(t))
    plt.bar(x - 0.2, t['ринок'], 0.4, label='Ефект ринку')
    plt.bar(x + 0.2, t['частка'], 0.4, label='Ефект частки ринку')
    plt.xticks(x, t['країна'], rotation=30)
    plt.title('Декомпозиція зміни продажів Renault 2020/2019 (топ-10 країн)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def correlation(df):
    cols = ['logTIV', 'MS', 'MS_грудень', 'ринок_ріст_%']
    c = df[cols].corr(method='spearman')
    print("\n--- Кореляційна матриця (Спірмен) ---")
    print(c.round(3))
    return c


def clustering(df, n_clusters=3):
    X = StandardScaler().fit_transform(df[['logTIV', 'MS', 'ринок_ріст_%']])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
    df['кластер'] = km.labels_
    print("\n--- Кластеризація країн (KMeans) ---")
    for k in range(n_clusters):
        g = df[df['кластер'] == k]
        print(f"кластер {k}: країн = {len(g):3d}  сер. ринок = {g['TIV'].mean():8.1f} тис.  "
              f"сер. MS = {g['MS'].mean():6.2f} %  ріст ринку = {g['ринок_ріст_%'].mean():6.1f} %  "
              f"приклади: {', '.join(g.sort_values('обсяг', ascending=False)['країна'].head(4))}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df['logTIV'], df['MS'], c=df['кластер'], cmap='viridis', s=40)
    for _, r in df.sort_values('обсяг', ascending=False).head(8).iterrows():
        plt.annotate(r['країна'], (r['logTIV'], r['MS']), fontsize=8)
    plt.xlabel('log10(ринок, авто)')
    plt.ylabel('частка Renault, %')
    plt.title('Кластери країн')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.hist(df['MS'], bins=25, color='steelblue')
    plt.title('Розподіл частки ринку Renault')
    plt.grid(True)
    plt.tight_layout()


def synthetic(df, n_sim=100):
    rng = np.random.default_rng(1)
    mu, sd = df['logTIV'].mean(), df['logTIV'].std()
    pos = df[df['MS'] > 0]
    p_zero = np.mean(df['MS'] == 0)
    a, b, _, _ = stats.beta.fit(pos['MS'] / 100, floc=0, fscale=1)
    ks = stats.kstest(pos['MS'] / 100, 'beta', args=(a, b)).pvalue
    print(f"\n--- Синтетична модель: logTIV ~ N({mu:.2f}, {sd:.2f}); MS = 0 з імовірністю {p_zero:.3f}, "
          f"інакше MS/100 ~ Beta({a:.3f}, {b:.3f}), КС-тест p = {ks:.3f} ---")

    res = []
    for _ in range(n_sim):
        lt = rng.normal(mu, sd, len(df))
        m = 100 * rng.beta(a, b, len(df))
        m[rng.random(len(df)) < p_zero] = 0
        res.append([lt.mean(), lt.std(), np.median(m), np.mean(m), stats.skew(m), stats.spearmanr(lt, m)[0]])
    res = np.array(res)
    real = [df['logTIV'].mean(), df['logTIV'].std(), df['MS'].median(), df['MS'].mean(), stats.skew(df['MS']),
            stats.spearmanr(df['logTIV'], df['MS'])[0]]
    names = ['сер. logTIV', 'СКВ logTIV', 'медіана MS', 'сер. MS', 'асиметрія MS', 'Спірмен(logTIV, MS)']
    print(f"{'показник':<22}{'реальні':>10}{'модель':>10}{'СКВ моделі':>12}")
    for n, r, s in zip(names, real, res.T):
        print(f"{n:<22}{r:10.3f}{s.mean():10.3f}{s.std():12.3f}")
