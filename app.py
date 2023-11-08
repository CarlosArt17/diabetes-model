from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Configuración de CORS
origins = ["*"]  # Esto permite el acceso desde cualquier origen. Ajusta según tus necesidades.

app = FastAPI(title='Diabetes Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Cargar el modelo previamente entrenado
model = load(pathlib.Path('model/diabetes-prediction-v1.joblib'))

# Definir las características necesarias en el mismo orden que en diabetes_features
required_features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Define una función para realizar one-hot encoding

# Modelo de datos de entrada
class InputData(BaseModel):
    gender:int= 0,
    age:float= 44.0,
    hypertension:int= 0,
    heart_disease:int= 0,
    smoking_history:int= 1,
    bmi:float= 19.31,
    HbA1c_level:float= 6.5,
    blood_glucose_level:int= 200

# Modelo de datos de respuesta
class OutputData(BaseModel):
    score:float=1.0

# Ruta para hacer predicciones
@app.post('/score', response_model=OutputData)
def score(data: InputData):
    # Validación de datos de entrada
    if data.age < 0 or data.bmi < 0:
        raise HTTPException(status_code=400, detail="Los valores de edad y BMI no pueden ser negativos.")

    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame(data.dict(), index=[0])

    # Reorganizar las columnas del DataFrame según las características requeridas
    input_data = input_data[required_features]


    # Asegurarse de que las columnas estén en el mismo orden que se usó para entrenar el modelo

    # Convertir los datos de entrada en un array numpy
    model_input = input_data.values.reshape(1, -1)

    # Realizar la predicción
    result = model.predict(model_input)

    return {'score': result[0]}
