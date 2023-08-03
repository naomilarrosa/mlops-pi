# Bibliotecas necesarias y creación de la aplicación FastAPI:
from fastapi import FastAPI

app = FastAPI()
# Creamos el DataFrame
import ast
import pandas as pd

rows = []
with open("steam_games.json") as f:
    for line in f.readlines():
        rows.append(ast.literal_eval(line))

df = pd.DataFrame(rows)
# Definimos una constante temporal para reemplazar los valores NA o NaN en la columna "release_date"
VALOR_TEMPORAL = "9999-01-01"

# Rellenar los valores NA o NaN en la columna "release_date" con el valor temporal
df["release_date"].fillna(VALOR_TEMPORAL, inplace=True)
# Función para obtener los 5 géneros más vendidos en un año
@app.get('/genero/')
def genero(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.startswith(Año)]
    
    # Obtener los géneros más vendidos en el año especificado
    top_generos = df_year['genres'].explode().value_counts().head(5).index.tolist()
    return top_generos
# Función para obtener los juegos lanzados en un año
@app.get('/juegos/')
def juegos(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.startswith(Año)]
    
    # Obtener los juegos lanzados en el año especificado
    juegos_lanzados = df_year['app_name'].tolist()
    return juegos_lanzados
# Función para obtener los 5 specs más repetidos en un año
@app.get('/specs/')
def specs(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.startswith(Año)]
    
    # Obtener los specs más repetidos en el año especificado
    top_specs = df_year['specs'].explode().value_counts().head(5).index.tolist()
    return top_specs
# Función para obtener la cantidad de juegos lanzados en un año con early access
import pandas as pd  # Asegúrate de importar pandas si aún no lo has hecho

# Suponiendo que el DataFrame "df" está definido globalmente o se pasa como un parámetro a la función

@app.get('/earlyacces/')
def earlyacces(Año: str):
    # Filtrar el DataFrame para el año especificado
    mask = (df['release_date'].str.startswith(Año, na=False)) & (df["early_access"] == True)
    df_year = df[mask]
    
    games = len(df_year)
    return {"games": games}

# Función para obtener el análisis de sentimiento por año
@app.get('/sentiment/')
def sentiment(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.startswith(Año)]
    
    # Obtener el análisis de sentimiento y contar la cantidad de registros en cada categoría
    analisis_sentimiento = df_year['sentiment'].value_counts().to_dict()
    return analisis_sentimiento
# Convertir la columna 'metascore' a un dtype numérico (si contiene valores numéricos)
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
# Función para obtener los top 5 juegos según el metascore en un año
@app.get('/metascore/')
def metascore(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.startswith(Año)]
    
    # Obtener los top 5 juegos con mayor metascore en el año especificado
    top_metascore_juegos = df_year.nlargest(5, 'metascore')['app_name'].tolist()
    return top_metascore_juegos


import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException, Form
from fastapi.templating import Jinja2Templates

# Cargar el modelo entrenado
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Definir una nueva clase para los parámetros de entrada
class InputData(BaseModel):
    early_access: int
    Action: int
    Adventure: int
    Simulation: int
    Strategy: int
    Indie: int
    Sports: int
    Philisophical: int

# Crear la aplicación FastAPI
app = FastAPI()

# Cargar las plantillas HTML
templates = Jinja2Templates(directory="templates")

# Ruta para mostrar el formulario de predicción
@app.get("/predict/")
async def show_prediction_form(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request})

# Ruta para hacer la predicción
@app.post("/predict/")
async def predict_price(
    early_access: int = Form(...),
    Action: int = Form(...),
    Adventure: int = Form(...),
    Simulation: int = Form(...),
    Strategy: int = Form(...),
    Indie: int = Form(...),
    Sports: int = Form(...),
    Philisophical: int = Form(...),
):
    # Crear un DataFrame con los datos de entrada
    data = {
        "early_access": [early_access],
        "Action": [Action],
        "Adventure": [Adventure],
        "Simulation": [Simulation],
        "Strategy": [Strategy],
        "Indie": [Indie],
        "Sports": [Sports],
        "Philisophical": [Philisophical]
    }
    input_df = pd.DataFrame(data)

    # Hacer la predicción utilizando el modelo cargado
    prediction = model.predict(input_df)

    # Devolver la predicción como resultado de la API
    return {"predicted_price": prediction[0]}
