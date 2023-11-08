import requests

body = {
    #"gender": 0,
    #"age": 80.0,
    #"hypertension": 0,
    #"heart_disease": 1,
    #"smoking_history": 1,
    #"bmi": 25.19,
    #"HbA1c_level": 6.6,
    #"blood_glucose_level": 140,
    "gender": 0,
    "age": 44.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": 1,
    "bmi": 19.31,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 200
}

try:
    response = requests.post(url='http://127.0.0.1:8000/score', json=body)

    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error en la solicitud. Código de estado: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Error en la solicitud: {e}")
