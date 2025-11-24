import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import openai

#Wczytanie danych
df = pd.read_parquet('/content/drive/MyDrive/ColabNotebooks/UNSW_NB15_training-set.parquet')

#Wybór cech i etykiet
X = df.drop(['label', 'attack_cat'], axis=1)
y = df['attack_cat']
#Przekształcenie zmiennych kategorycznych za pomocą kodowania one-hot
X_encoded = pd.get_dummies(X)
#Podział danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,
test_size=0.25, random_state=42)

#Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Ustawienie klucza API OpenAI
openai.api_key = 'chatgpt-openai-api-key'

#Funkcja do tworzenia promptów
def create_prompt(sample, label=None):
    sample_text = ", ".join(map(str, sample))
    prompt = f"Given the following network traffic data: {sample_text}\nIs this traffic normal or anomalous?"
    if label:
        prompt += f" It is {label}."
    return prompt

#Przygotowanie promptów treningowych (few-shot learning)
train_prompts = []
for i in range(len(X_train_scaled)):
    sample = X_train_scaled[i]
    label = y_train.iloc[i]
    prompt = create_prompt(sample, label)
    train_prompts.append(prompt)

#Przygotowanie promptów testowych
test_prompts = []
for i in range(len(X_test_scaled)):
    sample = X_test_scaled[i]
    prompt = create_prompt(sample)
    test_prompts.append(prompt)

#Klasyfikacja danych testowych
predictions = []
for prompt in test_prompts:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    prediction = response['choices'][0]['message']['content'].strip().lower()
    if "normal" in prediction:
        predictions.append("Normal")
    else:
        predictions.append("Anomalous")

#Ocena wyników
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")