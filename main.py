from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

app = FastAPI()

# Función para entrenar el modelo
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

# Entrenar el modelo al iniciar la aplicación (asegúrate de que el archivo CSV esté disponible)
if not os.path.exists('model.pkl'):
    train_model('Housing.csv')

# Cargar el modelo entrenado
model = joblib.load('model.pkl')

@app.get("/", response_class=HTMLResponse)
async def read_form():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Lab: Proyecto de Regresión Lineal de Machine Learning</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #000;
                color: #fff;
                margin: 0;
                padding: 0;
            }
            header {
                background-color: #FF6F00; /* Naranja */
                padding: 20px;
                text-align: center;
                color: #fff;
            }
            h1 {
                margin: 0;
                font-size: 2.5em;
            }
            h2 {
                font-size: 1.5em;
                margin-top: 10px;
            }
            .container {
                width: 80%;
                margin: 0 auto;
                padding: 20px;
                background-color: #222;
                border-radius: 8px;
            }
            label {
                display: block;
                margin: 10px 0 5px;
            }
            input {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                border: none;
                border-radius: 4px;
                background-color: #333;
                color: #fff;
            }
            button {
                padding: 10px 20px;
                background-color: #FF6F00; /* Naranja */
                color: #fff;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1em;
            }
            button:hover {
                background-color: #e65c00; /* Naranja oscuro */
            }
            .result {
                margin-top: 20px;
                font-size: 1.2em;
                color: #FF6F00; /* Naranja */
            }
        </style>
    </head>
    <body>
        <header>
            <h1>AI Lab</h1>
            <h2>Proyecto de Regresión Lineal de Machine Learning</h2>
        </header>
        <div class="container">
            <form id="inputForm">
                <label for="area">Área (en m²):</label>
                <input type="number" id="area" name="area" required>

                <label for="bedrooms">Habitaciones:</label>
                <input type="number" id="bedrooms" name="bedrooms" required>

                <label for="bathrooms">Baños:</label>
                <input type="number" id="bathrooms" name="bathrooms" required>

                <label for="stories">Niveles:</label>
                <input type="number" id="stories" name="stories" required>

                <label for="mainroad">¿Está en la carretera principal? (Sí/No):</label>
                <input type="text" id="mainroad" name="mainroad" required>

                <label for="guestroom">¿Tiene habitación de invitados? (Sí/No):</label>
                <input type="text" id="guestroom" name="guestroom" required>

                <label for="basement">¿Tiene sótano? (Sí/No):</label>
                <input type="text" id="basement" name="basement" required>

                <label for="hotwaterheating">¿Tiene calefacción de agua caliente? (Sí/No):</label>
                <input type="text" id="hotwaterheating" name="hotwaterheating" required>

                <label for="airconditioning">¿Tiene aire acondicionado? (Sí/No):</label>
                <input type="text" id="airconditioning" name="airconditioning" required>

                <label for="parking">Número de espacios de estacionamiento:</label>
                <input type="number" id="parking" name="parking" required>

                <label for="prefarea">¿Está en un área preferida? (Sí/No):</label>
                <input type="text" id="prefarea" name="prefarea" required>

                <label for="furnishingstatus">Estado de los muebles (Amueblado/Sin amueblar):</label>
                <input type="text" id="furnishingstatus" name="furnishingstatus" required>

                <button type="submit">Predecir Precio</button>
            </form>
            <div id="predictionResult" class="result"></div>
        </div>

        <script>
            document.getElementById('inputForm').onsubmit = async function(event) {
                event.preventDefault();

                const formData = new FormData(this);
                const response = await fetch('/predict/', {
                    method: 'POST',
                    body: formData,
                });
                const result = await response.json();
                document.getElementById('predictionResult').textContent = `Predicción del precio: $${result.prediction.toFixed(2)}`;
            };
        </script>
    </body>
    </html>
    """

@app.post("/predict/")
async def predict(area: float = Form(...), bedrooms: int = Form(...), bathrooms: int = Form(...), 
                    stories: int = Form(...), mainroad: str = Form(...), guestroom: str = Form(...), 
                    basement: str = Form(...), hotwaterheating: str = Form(...), 
                    airconditioning: str = Form(...), parking: int = Form(...), 
                    prefarea: str = Form(...), furnishingstatus: str = Form(...)):
    
    # Crear un DataFrame con los datos ingresados
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, 
                                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]],
                                columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                                        'guestroom', 'basement', 'hotwaterheating', 
                                        'airconditioning', 'parking', 'prefarea', 
                                        'furnishingstatus'])
    
    # Convertir variables categóricas a dummies
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Asegurarse de que las columnas coinciden con las del modelo
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0  # Rellenar columnas faltantes con ceros
    
    # Predecir el precio
    prediction = model.predict(input_data)
    
    return {"prediction": prediction[0]}

#Correr el programa con python -m uvicorn main:app --reload
