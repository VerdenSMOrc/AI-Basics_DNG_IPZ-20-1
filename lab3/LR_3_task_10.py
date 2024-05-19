import datetime
import json
import numpy as np
import matplotlib as plt
from sklearn import covariance, cluster
import yfinance as yf


# Завантаження прив'язок символів компаній до їх повних назв
input_file = 'company_symbol_mapping.json'
with open (input_file, 'r') as f:
    company_symbols_map = json.load(f)

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)
quotes = [yf.download(symbol, start=start_date, end=end_date,
                      ignore_tz=True) for symbol in symbols]

# Вилучення котирувань, що відповідають
# відкриттю та закриттю біржі
opening_quotes_temp = []
closing_quotes_temp = []

for quote in quotes:
    if 'Open' in quote and 'Close' in quote:
        opening_quotes_temp.append(quote['Open'].dropna().tolist())
        closing_quotes_temp.append(quote['Close'].dropna().tolist())

opening_quotes = []
for items in opening_quotes_temp:
    if len(items) != 0:
        opening_quotes.append(items)

closing_quotes = []
for items in closing_quotes_temp:
    if len(items) != 0:
        closing_quotes.append(items)

opening_quotes = np.array(opening_quotes).astype(np.float64)
closing_quotes = np.array(closing_quotes).astype(np.float64)

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()


for i in range(num_labels + 1):
    cluster_indices = np.where(labels == i)[0]
    cluster_names = [names[index] for index in cluster_indices if index < len(names)]
    print("Cluster", i + 1, "==>", ', '.join(cluster_names))
