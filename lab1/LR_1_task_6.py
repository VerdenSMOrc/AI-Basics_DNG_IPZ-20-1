from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Завантаження даних
with open("data_multivar_nb.txt", "r") as file:
    data = file.readlines()

# Підготовка даних
X = []
y = []
for line in data:
    line = line.strip().split(',')
    X.append([float(x) for x in line[:-1]])
    y.append(int(line[-1]))

# Розділення даних на навчальний та тестовий набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання моделей
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Прогнозування
svm_pred = svm_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

# Оцінка точності
svm_accuracy = accuracy_score(y_test, svm_pred)
nb_accuracy = accuracy_score(y_test, nb_pred)

# Вивід результатів
print("Точність машини опорних векторів:", svm_accuracy)
print("Точність наївного байєсівського класифікатора:", nb_accuracy)

if svm_accuracy > nb_accuracy:
    print("Точність машини опорних векторів більша.")
else:
    print("Точність наївного байєсівського класифікатора більша.")
