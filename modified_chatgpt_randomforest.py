#Inicjalizacja i trenowanie modelu Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_trainf, y_trainf['attack_cat'])
#Inicjalizacja i trenowanie modelu Random Forest
rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model2.fit(X_trainff, y_trainff['attack_cat'])
#Predykcja na zbiorze testowym
y_pred = rf_model.predict(X_testf)
# Predykcja na zbiorze testowym
y_pred2 = rf_model2.predict(X_testff)
#Ocena modelu
print("Confusion Matrix:")
print(confusion_matrix(y_testf['attack_cat'], y_pred))
print("\nClassification Report:")
print(classification_report(y_testf['attack_cat'], y_pred))
#Ocena modelu
print("Confusion Matrix:")
print(confusion_matrix(y_testff['attack_cat'], y_pred2))
print("\nClassification Report:")
print(classification_report(y_testff['attack_cat'], y_pred2))
