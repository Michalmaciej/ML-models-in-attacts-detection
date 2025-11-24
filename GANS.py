import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Wczytanie danych treningowych (wczytaj tylko część danych dla przyspieszenia)
dftrain = pd.read_parquet('/content/drive/MyDrive/ColabNotebooks/UNSW_NB15_testing-set.parquet')
#Ograniczenie do pierwszych 50000 wierszy
dftrain = dftrain.head(50000)
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
#Budowa modelu generatora
generator_input = Input(shape=(encoding_dim,))
x = Dense(64)(generator_input)
x = LeakyReLU(alpha=0.2)(x)
generator_output = Dense(input_dim, activation='sigmoid')(x)
generator = Model(generator_input, generator_output)

#Budowa modelu dyskryminatora
discriminator_input = Input(shape=(input_dim,))
y = Dense(64)(discriminator_input)
y = LeakyReLU(alpha=0.2)(y)
y = Dense(1, activation='sigmoid')(y)
discriminator = Model(discriminator_input, y)
discriminator.compile(optimizer='adam', loss='binary_crossentropy') #Dodana kompilacja

#Kompilacja modelu generatora
discriminator.trainable = False
gan_input = Input(shape=(encoding_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

#Trenowanie GAN
epochs = 50
batch_size = 256
for epoch in range(epochs):
    for _ in range(len(X_train_scaled) // batch_size):
        #Generowanie losowych punktów latentnych
        noise = np.random.normal(0, 1, size=(batch_size, encoding_dim))
        #Generowanie danych fałszywych z generatora
        generated_data = generator.predict(noise)
        #Losowanie próbek danych rzeczywistych
        idx = np.random.randint(0, X_train_scaled.shape[0], batch_size)
        real_data = X_train_scaled[idx]
        #Tworzenie zbioru treningowego dla dyskryminatora
        X = np.concatenate([real_data, generated_data])
        y_dis = np.zeros(2 * batch_size)
        y_dis[:batch_size] = 1 #Oznaczenie danych rzeczywistych jako 1
        #Trenowanie dyskryminatora
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)
        #Tworzenie zbioru treningowego dla GAN
        noise = np.random.normal(0, 1, size=(batch_size, encoding_dim))
        y_gan = np.ones(batch_size) #Oznaczenie wygenerowanych danych jako 1
        #Trenowanie GAN
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gan)
    print(f'Epoch: {epoch + 1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')

#Wykorzystanie generatora do generowania danych
generated_data = generator.predict(np.random.normal(0, 1, size=(len(X_test_scaled), encoding_dim)))
#Ocena jakości generowanych danych
mse = mean_squared_error(X_test_scaled, generated_data)
print(f'Test MSE: {mse}')
