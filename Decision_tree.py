from sklearn.tree import DecisionTreeClassifier
#Inicjalizacja i trenowanie modelu DTC
rf_model = DecisionTreeClassifier(random_state=42)
rf_model.fit(X_trainf, y_trainf['attack_cat'])
#Inicjalizacja i trenowanie modelu DTC
rf_model2 = DecisionTreeClassifier(random_state=42)
rf_model2.fit(X_trainff, y_trainff['attack_cat'])