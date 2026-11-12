import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ts_analysis import properties


def fit_model(x):
    d = np.diff(x)
    nz = d[d != 0]
    sigma = 1.4826 * np.median(np.abs(nz - np.median(nz)))
    jumps = nz[np.abs(nz - np.median(nz)) > 4 * sigma]
    small = nz[np.abs(nz - np.median(nz)) <= 4 * sigma]
    params = {
        'x0': x[0],
        'p_zero': np.mean(d == 0),
        'p_jump': len(jumps) / len(d),
        'mu': small.mean(),
        'sigma': small.std(),
        'jump_mu': jumps.mean(),
        'jump_sigma': jumps.std(),
    }
    print("\n--- Параметри синтетичної моделі ---")
    for k, v in params.items():
        print(f"{k:<12} {v:10.4f}")
    return params


def generate(params, n, rng):
    u = rng.random(n - 1)
    d = rng.normal(params['mu'], params['sigma'], n - 1)
    jump = u < params['p_jump']
    d[jump] = rng.normal(params['jump_mu'], params['jump_sigma'], jump.sum())
    zero = (u >= params['p_jump']) & (u < params['p_jump'] + params['p_zero'])
    d[zero] = 0
    return params['x0'] + np.concatenate([[0], np.cumsum(d)])


def compare(x, params, n_sim=100):
    rng = np.random.default_rng(1)
    sims = [generate(params, len(x), rng) for _ in range(n_sim)]
    real = properties(x)
    sim_props = [properties(s) for s in sims]

    print(f"\n--- Порівняння властивостей: реальні дані vs модель ({n_sim} реалізацій) ---")
    print(f"{'показник':<24}{'реальні':>10}{'модель':>10}{'СКВ моделі':>12}")
    for k in real:
        v = np.array([p[k] for p in sim_props])
        print(f"{k:<24}{real[k]:10.4f}{v.mean():10.4f}{v.std():12.4f}")

    ks = stats.ks_2samp(np.diff(x), np.diff(sims[0]))
    print(f"\nКС-тест розподілу щоденних змін: статистика = {ks.statistic:.4f}, p-value = {ks.pvalue:.4f}")

    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(x, color='black', linewidth=2, label='Реальний курс купівлі')
    for i in range(3):
        plt.plot(sims[i], alpha=0.8, label=f'Синтетична реалізація {i + 1}')
    plt.title('Реальний та синтетичні часові ряди')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.hist(np.diff(x), bins=50, alpha=0.6, density=True, label='Реальні зміни')
    plt.hist(np.concatenate([np.diff(s) for s in sims]), bins=50, alpha=0.6, density=True, range=(np.diff(x).min(), np.diff(x).max()),
             label='Зміни в моделі')
    plt.title('Розподіл щоденних змін')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return sims
