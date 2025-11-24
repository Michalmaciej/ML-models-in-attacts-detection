import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#Wczytanie danych treningowych
dftrain = pd.read_parquet('/content/drive/MyDrive/ColabNotebooks/UNSW_NB15_training-set.parquet')

#Usunięcie kolumn 'label' i 'attack_cat' z danych treningowych
X_train = dftrain.drop(['label', 'attack_cat'], axis=1)

#Przekształcenie zmiennych kategorycznych za pomocą kodowania one-hot
X_train_encoded = pd.get_dummies(X_train)

#Podział danych na zbiory treningowe i testowe
X_trainf, X_testf = train_test_split(X_train_encoded, test_size=0.25, random_state=42)

#Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_trainf)
X_test_scaled = scaler.transform(X_testf)

#Parametry modelu
input_dim = X_train_scaled.shape[1]
encoding_dim = 32 #Rozmiar warstwy latentnej

#Budowa modelu autoenkodera
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

#Kompilacja modelu
autoencoder.compile(optimizer='adam', loss='mse')

#Trenowanie modelu
autoencoder.fit(X_train_scaled, X_train_scaled,
epochs=50,
batch_size=256,
shuffle=True,
validation_data=(X_test_scaled, X_test_scaled))

#Wykorzystanie enkodera do kodowania danych
encoded_train = encoder.predict(X_train_scaled)
encoded_test = encoder.predict(X_test_scaled)
#Wykorzystanie dekodera do dekodowania danych
decoded_train = decoder.predict(encoded_train)
decoded_test = decoder.predict(encoded_test)

#Ocena jakości rekonstrukcji
train_mse = mean_squared_error(X_train_scaled, decoded_train)
test_mse = mean_squared_error(X_test_scaled, decoded_test)
print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')