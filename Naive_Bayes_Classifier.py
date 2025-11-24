from sklearn.naive_bayes import GaussianNB
#Inicjalizacja i trenowanie modelu NB
rf_model = GaussianNB()
rf_model.fit(X_trainf, y_trainf['attack_cat'])
#Inicjalizacja i trenowanie modelu NB
rf_model2 = GaussianNB()
rf_model2.fit(X_trainff, y_trainff['attack_cat'])