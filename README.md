

# <h1 align=center> **Proyecto: Machine Learning Operations (MLOps)** </h1>
                                            

<p align="center">
<img src="https://raw.githubusercontent.com/MatyTrova/PI-MLOps/main/imgs/mlops.png"  height=300>
</p>

--- 
## `Descripción`

Este proyecto tiene como objetivo desarrollar el rol de `Data Scientist` y `Data Engineer`, aplicando técnicas de extracción, transformación y carga de datos (`ETL`), análisis exploratorio de datos (`EDA`) y creación de un sistema de predicción de precio de los juegos de steam basado en machine learning.

Para ello, se utilizará un set de datos de plataformas de steam, con el fin de explorar, entender los patrones  y así generar predicciones en los precios.

El proyecto abarca desde la extracción de los datos, hasta la implementación del sistema de predicción. Asimismo, esta documentada cada etapa del proceso en cada archivo para la realización del modelo de `machine learning`.

Finalmente, se desplegará el proyecto como una `API` virtual en la plataforma de la nube de Render, lo que permitirá el acceso a los resultados desde cualquier lugar y dispositivo.

Este proyecto es una oportunidad para explorar en profundidad el proceso de desarrollo de un sistema de predicción y las herramientas utilizadas en el camino, así como para aprender sobre el manejo y análisis de datos en el contexto de la plataforma de juegos.

Nos proponían empezar desde cero haciendo un trabajo rápido de data engineer y tener un MVP (Minimum Viable Product), para el cierre del proyecto. Realizando una API REST con 6 funciones:

`def genero( Año: str ): Se ingresa un año y devuelve una lista con los 5 géneros más ofrecidos en el orden correspondiente.`

`def juegos( Año: str ): Se ingresa un año y devuelve una lista con los juegos lanzados en el año.`

`def specs( Año: str ): Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.`

`def earlyacces( Año: str ): Cantidad de juegos lanzados en un año con early access.`

`def sentiment( Año: str ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.`

`def metascore( Año: str ): Top 5 juegos según año con mayor metascore.`

Y por último, crear un `modelo de predicción` en el que, con las variables elejidas (metascore, año y género), deberíamos predecir el precio del juego y el RMSE del modelo.
## `API`

A continuación estará el enlace a la respectiva api y su documentación: 
+ https://mlops-pi-q2g7.onrender.com/
+ https://mlops-pi-q2g7.onrender.com/docs

Además de un video explicativo acerca de esta : 

+ https://www.youtube.com/watch?v=_i6Ku3UrnOQ

## `Estructura del repositorio`

+ `README.md`: Archivo principal con información detallada del proyecto.

+ `modelo.ipynb`: Contiene todo el desarrollo del modelo de machine learning para la predicción de precios de los juegos.

+ `EDA.ipynb`: Contiene el código para el análisis exploratorio de datos.

+ `main.py`: Contiene todo el código, la formación y correcto funcionamiento de la API.

+ `requirements.txt`: Archivo con las dependencias y librerías necesarias para ejecutar el proyecto.
+ `model.pkl`: Archivo en el cual guarde el modelo entrenado para luego cargarlo y usarlo en main.py .
+ `steam_games.json`: dataset con el cual realicé el modelo.

El proceso de `ETL` esta con código comentado en modelo.ipynb y tambien algunas transformaciones en main.py para el correcto funcionamiento.


##### Nota :Para acceder a todos los datasets utilizados se puede ingresar al siguiente enlace de drive : 

https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj



<p align="center">
<img src="https://raw.githubusercontent.com/MatyTrova/PI-MLOps/main/imgs/henry.jpg"  alt="MLOps">
</p>
