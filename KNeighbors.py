from sklearn.neighbors import KNeighborsClassifier
#Inicjalizacja i trenowanie modelu KNN
rf_model = KNeighborsClassifier()
rf_model.fit(X_trainf, y_trainf['attack_cat'])
#Inicjalizacja i trenowanie modelu KNN
rf_model2 = KNeighborsClassifier()
rf_model2.fit(X_trainff, y_trainff['attack_cat'])