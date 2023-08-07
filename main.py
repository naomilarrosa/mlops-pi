# Bibliotecas necesarias y creación de la aplicación FastAPI:
from fastapi import FastAPI
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from enum import Enum
import ast
app = FastAPI(title="Proyecto MLOps By Naomi")
# Creamos el DataFrame
import ast

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
    df_year = df[df['release_date'].str.contains(Año, case=False, na=False)]
    
    # Obtener el conteo de géneros en el año especificado
    conteo_generos = df_year['genres'].explode().value_counts().to_dict()
     # Obtener solo los 5 géneros más vendidos
    top_5_generos = dict(sorted(conteo_generos.items(), key=lambda item: item[1], reverse=True)[:5])
    
    return top_5_generos

# Función para obtener los juegos lanzados en un año
@app.get('/juegos/')
def juegos(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.contains(Año)]
    # Obtener los juegos lanzados en el año especificado
    juegos_lanzados = df_year['app_name'].tolist()
    return {Año: juegos_lanzados}
# Función para obtener los 5 specs más repetidos en un año
@app.get('/specs/')
def specs(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.contains(Año)]
    
    # Obtener los specs más repetidos en el año especificado
    top_specs = df_year['specs'].explode().value_counts().head(5).index.tolist()
    return {Año: top_specs}
# Función para obtener la cantidad de juegos lanzados en un año con early access
@app.get('/earlyacces/')
def earlyacces(Año: str):
    # Filtrar el DataFrame para el año especificado
    mask = (df['release_date'].str.contains(Año, na=False)) & (df["early_access"] == True)
    df_year = df[mask]
    
    games = len(df_year)
    return {"Cantidad de Juegos": games}

# Suponiendo que el DataFrame "df" está definido globalmente o se pasa como un parámetro a la función

# Crear una lista con los valores de sentimientos válidos
sentiments = ['Overwhelmingly Positive', 'Very Positive', 'Positive', 'Mostly Positive',
 'Mixed', 'Mostly Negative', 'Negative', 'Very Negative',
 'Overwhelmingly Negative']

# Filtrar el DataFrame para quedarse solo con las filas que tengan esos valores en la columna 'sentiment'
df = df[df['sentiment'].isin(sentiments)]

# Función para obtener el análisis de sentimiento por año
@app.get('/sentiment/')
def sentiment(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.contains(Año)]
    
    # Obtener el análisis de sentimiento y contar la cantidad de registros en cada categoría
    analisis_sentimiento = df_year['sentiment'].value_counts().to_dict()
    
    # Imprimir los valores disponibles de sentimientos
    print(df['sentiment'].unique())
    
    return analisis_sentimiento
# Convertir la columna 'metascore' a un dtype numérico (si contiene valores numéricos)
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
# Función para obtener los top 5 juegos según el metascore en un año
@app.get('/metascore/')
def metascore(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.contains(Año)]
    
    # Obtener los top 5 juegos con mayor metascore en el año especificado
    top_metascore_juegos = df_year.nlargest(5, 'metascore')[['app_name', 'metascore']].to_dict('records')
    return top_metascore_juegos

# Cargar el modelo pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Crear el Enum de géneros
class Genre(Enum):
    Action = "Action"
    Adventure = "Adventure"
    Casual = "Casual"
    Early_Access = "Early Access"
    Free_to_Play = "Free to Play"
    Indie = "Indie"
    Massively_Multiplayer = "Massively Multiplayer"
    RPG = "RPG"
    Racing = "Racing"
    Simulation = "Simulation"
    Sports = "Sports"
    Strategy = "Strategy"
    Video_Production = "Video Production"

# Definir la ruta de predicción
@app.get("/predicción") 
def predict(metascore: float = None, earlyaccess: bool = None, Año: str = None, genero: Genre = None):
    # Validar que se hayan pasado los parámetros necesarios
    if metascore is None or Año is None or genero is None or earlyaccess is None:
        raise HTTPException(status_code=400, detail="Missing parameters")
    
    # Convertir el input en un DataFrame con las columnas necesarias para el modelo
    input_df = pd.DataFrame([[metascore, earlyaccess, Año, *[1 if genero.value == g else 0 for g in Genre._member_names_]]], columns=['metascore', 'year', 'early_access', *Genre._member_names_])
    
    # Verificar si el género es Free to Play
    if genero == Genre.Free_to_Play:
        # Devolver 0 como precio
        return {"price": 0, "RMSE del modelo": 8.36}
    else:
        # Realizar la predicción con el modelo
        try:
            price = model.predict(input_df)[0]
        except:
            raise HTTPException(status_code=400, detail="Invalid input")

        # Devolver el precio y el RMSE como salida
        return {"price": price, "RMSE del modelo": 8.36}