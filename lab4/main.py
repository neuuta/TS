import warnings
import matplotlib.pyplot as plt
import ts_analysis as ta
import ts_model as tm
import dataset_analysis as da

warnings.filterwarnings('ignore')

print("========== ГРУПА ВИМОГ 1: курс USD (Ощадбанк) ==========")
df = ta.load_real_data('data/Oschadbank (USD).xls')
ts = df['Купівля']
print(f"Період: {df.index[0].date()} - {df.index[-1].date()}, вимірів: {len(df)}")

ta.plot_real(df)
ta.print_properties('курс купівлі (реальні дані)', ta.properties(ts.values))
ta.decomposition(ts)
ta.correlation(df)
ta.clustering(ts)

params = tm.fit_model(ts.values)
tm.compare(ts.values, params)

print("\n========== ГРУПА ВИМОГ 2: Data_Set_1 ==========")
ds = da.load_dataset('data/Data_Set_1.xlsx')
da.describe(ds)
da.decomposition(ds)
da.correlation(ds)
da.clustering(ds)
da.synthetic(ds)

plt.show()
