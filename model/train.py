import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import dump

# Cargar el conjunto de datos
data_path = pathlib.Path('data/diabetes_prediction.csv')
df = pd.read_csv(data_path)

# Definir las características necesarias en el mismo orden que en diabetes_features
diabetes_features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']

# Reorganizar las columnas del DataFrame según diabetes_features
df = df[diabetes_features]

# Separar características (X) y variable objetivo (y)
y = 1 - df.pop('diabetes')
X = df

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenar el modelo RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
diabetes_preds = forest_model.predict(X_test)

# Calcular el error medio absoluto
mae = mean_absolute_error(y_test, diabetes_preds)
print("Mean Absolute Error:", mae)

# Guardar el modelo entrenado
model_path = pathlib.Path('model/diabetes-prediction-v1.joblib')
dump(forest_model, model_path)

print('Modelo entrenado y guardado en', model_path)
