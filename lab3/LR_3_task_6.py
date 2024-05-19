import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Генерація даних
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 2 * np.sin(X).flatten() + np.random.uniform(-0.5, 0.5, m)


# Функція для побудови кривих навчання
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


# Побудова моделі поліноміальної регресії
polynomial_regressor = Pipeline([("poly_features",
                                     PolynomialFeatures(degree=10, include_bias=False)),
                                 ("lin_reg", LinearRegression())])


polynomial_regressor_second = Pipeline([("poly_features",
                                     PolynomialFeatures(degree=2, include_bias=False)),
                                 ("lin_reg", LinearRegression())])


# Візуалізація кривих навчання
plot_learning_curves(LinearRegression(), X, y)
plt.show()
plot_learning_curves(polynomial_regressor, X, y)
plt.show()
plot_learning_curves(polynomial_regressor_second, X, y)
plt.show()
