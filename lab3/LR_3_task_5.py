import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Генерація випадкових даних
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 2 * np.sin(X).flatten() + np.random.uniform(-0.5, 0.5, m)

# Побудова моделі лінійної регресії
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)

# Побудова моделі поліноміальної регресії
degree = 5
polynomial_features = PolynomialFeatures(degree=degree)
polynomial_regressor = make_pipeline(polynomial_features, LinearRegression())
polynomial_regressor.fit(X, y)
y_pred_poly = polynomial_regressor.predict(X)

# Виведення графіків
plt.scatter(X, y, color='red', label='Original data')
plt.plot(X, y_pred_linear, color='blue', label='Linear regression')
plt.plot(X, y_pred_poly, color='green', label='Polynomial regression')
plt.legend()
plt.show()

# Оцінка якості моделей
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

print(f"Лінійна регресія: MSE = {mse_linear:.2f}, R2 = {r2_linear:.2f}")
print(f"Поліноміальна регресія: MSE = {mse_poly:.2f}, R2 = {r2_poly:.2f}")
