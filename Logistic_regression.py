from sklearn.linear_model import LogisticRegression
#Inicjalizacja i trenowanie modelu LR
rf_model = LogisticRegression(
random_state=42, #Ustawienie ziarna losowości dla powtarzalności wyników
max_iter=1000, #Zwiększenie maksymalnej liczby iteracji dla zapewnienia zbieżności
solver='lbfgs', # stawienie solvera na 'lbfgs' (Algorytm Quasi-Newtona)
multi_class='auto' #Określenie metody dla wielu klas ('auto' automatycznie wybiera metodę)
)
rf_model.fit(X_trainf, y_trainf['attack_cat'])
#Inicjalizacja i trenowanie modelu LR
rf_model2 = LogisticRegression(
random_state=42, #Ustawienie ziarna losowości dla powtarzalności wyników
max_iter=1000, #Zwiększenie maksymalnej liczby iteracji dla zapewnieniazbieżności
solver='lbfgs', #Ustawienie solvera na 'lbfgs' (Algorytm Quasi-Newtona)
multi_class='auto' #Określenie metody dla wielu klas ('auto' automatycznie wybiera metodę)
)
rf_model2.fit(X_trainff, y_trainff['attack_cat'])