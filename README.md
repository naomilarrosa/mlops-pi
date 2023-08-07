

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

Y por último, crear un `modelo de predicción` en el que, con las variables elejidas (metascore, earlyaccess(acceso temprano), año y género), deberíamos predecir el precio del juego y el RMSE del modelo.
## `API`

A continuación estará el enlace a la respectiva api y su documentación: 
+ https://mlops-pi-q2g7.onrender.com
+ https://mlops-pi-q2g7.onrender.com/docs

Además de un video explicativo acerca de esta : 

+ En proceso..

## Deployment

 El despliegue de la API fue realizado mediante render.

 Aquí el tutorial utilizado: https://github.com/HX-FNegrete/render-fastapi-tutorial 

## `Estructura del repositorio`

+ `README.md`: Archivo principal con información detallada del proyecto.

+ `modelo.ipynb`: Contiene todo el desarrollo del modelo de machine learning para la predicción de precios de los juegos.

+ `EDA.ipynb`: Contiene el código para el análisis exploratorio de datos.

+ `main.py`: Contiene todo el código, la formación y correcto funcionamiento de la API.

+ `requirements.txt`: Archivo con las dependencias y librerías necesarias para ejecutar el proyecto.
+ `model.pkl`: Archivo en el cual guarde el modelo entrenado para luego cargarlo y usarlo en main.py .
+ `steam_games.json`: dataset con el cual realicé el modelo.

El proceso de `ETL` esta con código comentado en modelo.ipynb y tambien algunas transformaciones en main.py para el correcto funcionamiento.

## Tecnologia usada
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
- Pikle
- Enum
##### Nota :Para acceder a todos los datasets utilizados se puede ingresar al siguiente enlace de drive : 

https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj



<p align="center">
<img src="https://raw.githubusercontent.com/MatyTrova/PI-MLOps/main/imgs/henry.jpg"  alt="MLOps">
</p>
