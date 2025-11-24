from sklearn.neural_network import MLPClassifier
#Inicjalizacja i trenowanie modelu MLPC
rf_model = MLPClassifier(random_state=42)
rf_model.fit(X_trainf, y_trainf['attack_cat'])
#Inicjalizacja i trenowanie modelu MLPC
rf_model2 = MLPClassifier(random_state=42)
rf_model2.fit(X_trainff, y_trainff['attack_cat'])
