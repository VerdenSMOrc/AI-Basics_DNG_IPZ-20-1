import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантаження даних
url = "income_data.txt"
names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital-Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss', 'Hours-per-week', 'Country', 'Income']
data = pd.read_csv(url, names=names)

# Перетворення категорійних даних у числові
labelEncoder = LabelEncoder()
for column in data.columns:
    data[column] = labelEncoder.fit_transform(data[column])

# Розділення даних на атрибути та мітки
X = data.drop('Income', axis=1)
y = data['Income']

# Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Список моделей для оцінювання
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Оцінювання моделей
results = {}
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    results[name] = (accuracy, report)

# Виведення результатів
for name, (accuracy, report) in results.items():
    print(f"Модель: {name}")
    print(f"Точність: {accuracy:.2f}")
    print(report)
