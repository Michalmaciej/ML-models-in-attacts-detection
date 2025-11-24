import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#Wczytanie danych treningowych i testowych
dftrain = pd.read_parquet('/content/drive/MyDrive/Colab
Notebooks/UNSW_NB15_training-set.parquet')
dftest = pd.read_parquet('/content/drive/MyDrive/Colab
Notebooks/UNSW_NB15_testing-set.parquet')
#Usunięcie kolumn 'label' i 'attack_cat' z danych treningowych
X_train = dftrain.drop(['label', 'attack_cat'], axis=1)
#Kolumny 'label' i 'attack_cat' jako etykiety dla danych treningowych
y_train = dftrain[['label', 'attack_cat']]
#Usunięcie kolumn 'label' i 'attack_cat' z danych testowych
X_test = dftest.drop(['label', 'attack_cat'], axis=1)
#Kolumny 'label' i 'attack_cat' jako etykiety dla danych testowych
y_test = dftest[['label', 'attack_cat']]
#Przekształcenie zmiennych kategorycznych za pomocą kodowania one-hot
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)
X_trainf, X_testf, y_trainf, y_testf = train_test_split(X_train_encoded, y_train, test_size=0.25, random_state=42)
X_trainff, X_testff, y_trainff, y_testff = train_test_split(X_test_encoded, y_test, test_size=0.25, random_state=42)
#Inicjalizacja i trenowanie modelu Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_trainf, y_trainf['label'])
#Inicjalizacja i trenowanie modelu Random Forest
rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model2.fit(X_trainff, y_trainff['label'])
#Predykcja na zbiorze testowym
y_pred = rf_model.predict(X_testf)
#Predykcja na zbiorze testowym
y_pred2 = rf_model2.predict(X_testff)
#Ocena modelu
print("Confusion Matrix:")
print(confusion_matrix(y_testf['label'], y_pred))
print("\nClassification Report:")
print(classification_report(y_testf['label'], y_pred))
#Ocena modelu
print("Confusion Matrix:")
print(confusion_matrix(y_testff['label'], y_pred2))
print("\nClassification Report:")
print(classification_report(y_testff['label'], y_pred2))
