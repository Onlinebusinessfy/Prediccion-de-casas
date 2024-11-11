# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model(csv_file: str):
    df = pd.read_csv(csv_file)
    
    # Variables independientes y dependientes
    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
            'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
            'parking', 'prefarea', 'furnishingstatus']]  # Características
    y = df['price']  # Variable dependiente (precio)
    
    # Convertir variables categóricas a dummies
    X = pd.get_dummies(X, drop_first=True)
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo de regresión lineal
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Guardar el modelo entrenado
    joblib.dump(reg, 'model.pkl')
    
    return mse

def load_model():
    """
    Carga el modelo entrenado desde un archivo.
    """
    return joblib.load('model.pkl')