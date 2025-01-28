# Series Temporales: Predicción de demanda en aplicaciones de taxis.
"Para una compañía de taxis, predecir la cantidad de viajes en la próxima hora para que la compañía pueda congregar más conductores durante las horas pico."

En este proyecto se analiza un dataset compuesto únicamente de la etiqueta temporal y la cantidad de órdenes, se aplica limpieza de datos y se emplean técnicas de análisis como re-muestreo, media móvil, descomposición en tendencia, estacionalidad y residuo, al igual que retrasos para generar características que permitan entrenar múltiples modelos de machine learning.

# Descripción del proyecto
La compañía Sweet Lift Taxi ha recopilado datos históricos sobre pedidos de taxis en los aeropuertos. Para atraer a más conductores durante las horas pico, necesitamos predecir la cantidad de pedidos de taxis para la próxima hora. Construye un modelo para dicha predicción.

La métrica RECM en el conjunto de prueba no debe ser superior a 48.

Instrucciones del proyecto
Descarga los datos y remuestréalos de tal forma que cada punto de datos de los datos originales caigan dentro de intervalos de una hora.
Analiza los datos.
Entrena diferentes modelos con diferentes hiperparámetros. La muestra de prueba debe ser el 10% del conjunto de datos inicial.
Prueba los datos usando la muestra de prueba y proporciona una conclusión.
Descripción de datos
Los datos se almacenan en el archivo /datasets/taxi.csv.  Descarga el conjunto de datos

El número de pedidos está en la columna num_orders.
