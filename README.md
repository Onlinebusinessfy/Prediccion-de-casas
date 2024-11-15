# AI Lab: Proyecto de Regresión Lineal de Machine Learning

## Descripción del Proyecto

Este proyecto tiene como objetivo predecir el precio de una casa basado en características como el área, número de habitaciones, nivel de la casa, entre otras. Utiliza un modelo de regresión lineal entrenado con datos de propiedades inmobiliarias y se integra con una aplicación web construida con **FastAPI** y **HTML**.

---

## Estructura del Proyecto


---

## Requisitos

Antes de comenzar, asegúrate de tener **Python 3.7+** instalado. Además, instala las siguientes dependencias:


### Dependencias

- `fastapi`: Framework web de alto rendimiento para construir APIs.
- `pandas`: Librería para manipulación de datos.
- `scikit-learn`: Herramienta para modelado y machine learning.
- `joblib`: Utilizado para guardar y cargar modelos entrenados.
- `uvicorn`: Servidor ASGI para FastAPI.

---

## Instrucciones para Entrenar el Modelo

1. Coloca tu archivo CSV con los datos de las casas en el directorio del proyecto.
2. En el archivo `model.py`, asegúrate de que el archivo CSV sea el adecuado para las predicciones.
3. Ejecuta el siguiente código para entrenar el modelo:

```python
from model import train_model

# Entrenar el modelo con tu archivo CSV
mse = train_model('archivo.csv')
print(f"Error cuadrático medio del modelo: {mse}")

Correr el modelo con
uvicorn main:app --reload
o
python -m uvicorn main:app --reload

Explicación del Código
model.py: Este archivo contiene la función para entrenar el modelo de regresión lineal. También incluye la función para cargar el modelo previamente entrenado (model.pkl).

app.py: Este archivo contiene el código de la aplicación web en FastAPI que maneja las peticiones POST para recibir los datos del formulario y devolver la predicción.

index.html: La interfaz web donde el usuario puede ingresar los detalles de la casa. El formulario envía estos datos a la aplicación FastAPI para obtener el precio estimado.

Cómo Funciona
El usuario llena un formulario con características de la casa, como área, número de habitaciones, número de baños, etc.
Los datos del formulario se envían al servidor FastAPI, que pasa los valores a un modelo de regresión lineal.
El modelo predice el precio de la casa basado en los valores proporcionados y devuelve la predicción al usuario.
