import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split


# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# --------------------------------

# Розділення даних на навчальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення SVM-класифікатора з поліноміальним ядром
classifier_poly = SVC(kernel='poly', degree=8, random_state=0)

# Навчання класифікатора
classifier_poly.fit(X_train, y_train)

# Обчислення показників якості
y_test_pred_poly = classifier_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_test_pred_poly)
recall_poly = recall_score(y_test, y_test_pred_poly, average='weighted')
precision_poly = precision_score(y_test, y_test_pred_poly, average='weighted')

# Виведення показників якості
print("Поліноміальне ядро:")
print(f"Акуратність: {accuracy_poly:.2f}")
print(f"Повнота: {recall_poly:.2f}")
print(f"Точність: {precision_poly:.2f}")