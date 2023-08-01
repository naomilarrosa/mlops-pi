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
@app.get('/earlyacces/')
def earlyacces(Año: str):
    # Filtrar el DataFrame para el año especificado
    df_year = df[df['release_date'].str.startswith(Año)]
    
    # Contar la cantidad de juegos con early access en el año especificado
    cantidad_early_access = df_year['early_access'].sum()
    return cantidad_early_access
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


