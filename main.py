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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

class Item(BaseModel):
    early_access: int
    Adventure: float
    Action: float
    Indie: float
    Casual: float
    Simulation: float
    Strategy: float
    RPG: float
    Sports: float
    Racing: float
    Massively_Multiplayer: float
    Animation_Modeling: float
    Video_Production: float
    Utilities: float
    Web_Publishing: float
    Education: float
    Software_Training: float
    Design_Illustration: float
    Audio_Production: float
    Photo_Editing: float
    Accounting: float

def load_model_from_pickle():
    # Cargar el modelo desde el archivo pickle
    with open('model.pkl', 'rb') as file:
        modelo_entrenado = pickle.load(file)
    return modelo_entrenado

def predict_price(input_data):
    # Cargar el modelo desde el archivo pickle
    model = load_model_from_pickle()

    # Realizar la predicción
    prediction = model.predict(input_data)

    # Tomar el valor de la predicción
    prediction_value = prediction[0]

    return prediction_value

@app.post("/predict_price/")
async def get_prediction(item: Item):
    try:
        input_data = [list(item.dict().values())]
        prediction = predict_price(input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
