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

# Створення SVM-класифікатора з гаусовим ядром
classifier_rbf = SVC(kernel='rbf', random_state=0)

# Навчання класифікатора
classifier_rbf.fit(X_train, y_train)

# Обчислення показників якості
y_test_pred_rbf = classifier_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_test_pred_rbf)
recall_rbf = recall_score(y_test, y_test_pred_rbf, average='weighted')
precision_rbf = precision_score(y_test, y_test_pred_rbf, average='weighted')

# Виведення показників якості
print("Гаусове ядро:")
print(f"Акуратність: {accuracy_rbf:.2f}")
print(f"Повнота: {recall_rbf:.2f}")
print(f"Точність: {precision_rbf:.2f}")
