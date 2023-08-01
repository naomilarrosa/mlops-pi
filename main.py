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


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import ast
import pandas as pd



rows = []
with open("steam_games.json") as f:
    for line in f.readlines():
        rows.append(ast.literal_eval(line))

df = pd.DataFrame(rows)
# Convertir la columna "release_date" a formato de fecha y extraer el año
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year

# Codificar la variable categórica "genre"
label_encoder = LabelEncoder()
df = df.dropna(subset=['genres'])  # Eliminar filas con valores 'float' en la columna 'genres'
df['genres'] = df['genres'].apply(lambda x: ', '.join(x))  # Convertir listas a cadenas
df['genres_encoded'] = label_encoder.fit_transform(df['genres'])
df.drop('genres', axis=1, inplace=True)

# Convertir la columna 'metascore' a valores numéricos y reemplazar los valores 'nan' y 'na' por NaN
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')

# Tratar los valores 'Free to Play' en la columna 'price' como 0
df.loc[df['price'] == 'Free to Play', 'price'] = 0

# Convertir la columna 'price' a valores numéricos y reemplazar los valores 'nan' por NaN
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Eliminar filas que contienen NaN en cualquier columna excepto 'price'
df.dropna(subset=['genres_encoded', 'year', 'metascore'], inplace=True)

# Eliminar filas que contienen NaN en la columna 'price'
df.dropna(subset=['price'], inplace=True)

# Seleccionar las características para el modelo
X = df[['genres_encoded', 'year', 'metascore']]
y = df['price']  # Variable objetivo que deseamos predecir (precio)

# Aplicar SimpleImputer para reemplazar los valores NaN en las características restantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión
model = LinearRegression()
model.fit(X_train, y_train)  # Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)  # Calcular el error cuadrático medio (RMC)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Error Cuadrático Medio (RMC):", rmse)

# Cargar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

class PredictionInput(BaseModel):
    genero: str
    earlyaccess: bool
    year: int
    metascore: float


@app.post('/prediccion/')
def predicción(input_data: PredictionInput):
    # Convertir el input en un DataFrame para hacer la predicción
    input_df = pd.DataFrame([input_data.dict()])

    # Codificar la variable categórica "genero"
    input_df['genero_encoded'] = label_encoder.transform([input_data.genero])

    # Seleccionar las características para la predicción
    X_pred = input_df[['genero_encoded', 'year', 'metascore']]
    
    # Aplicar SimpleImputer para reemplazar los valores NaN en las características
    X_pred_imputed = imputer.transform(X_pred)

    # Realizar la predicción con el modelo
    predicted_price = model.predict(X_pred_imputed)[0]

    # Agregar el precio predicho al resultado de la predicción
    prediction_result = input_data.dict()
    prediction_result['predicted_price'] = predicted_price

    return prediction_result
