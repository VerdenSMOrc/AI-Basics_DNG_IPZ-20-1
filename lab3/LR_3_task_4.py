import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()

regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

# Коефіцієнти регресії
print("Коефіцієнти регресії:", regr.coef_)
# Перехоплення (intercept)
print("Перехоплення:", regr.intercept_)
# R^2 (коефіцієнт детермінації)
print("R^2:", r2_score(ytest, ypred))
# Середня абсолютна помилка
print("Середня абсолютна помилка:", mean_absolute_error(ytest, ypred))
# Середньоквадратична помилка
print("Середньоквадратична помилка:", mean_squared_error(ytest, ypred))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()