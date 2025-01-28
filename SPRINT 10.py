#!/usr/bin/env python
# coding: utf-8

# Hola Pia! Como te va?
# 
# Mi nombre es Facundo! Un gusto conocerte, seré tu revisor en este proyecto.
# 
# A continuación un poco sobre la modalidad de revisión que usaremos:
# 
# Cuando enccuentro un error por primera vez, simplemente lo señalaré, te dejaré encontrarlo y arreglarlo tú cuenta. Además, a lo largo del texto iré haciendo algunas observaciones sobre mejora en tu código y también haré comentarios sobre tus percepciones sobre el tema. Pero si aún no puedes realizar esta tarea, te daré una pista más precisa en la próxima iteración y también algunos ejemplos prácticos. Estaré abierto a comentarios y discusiones sobre el tema.
# 
# Encontrará mis comentarios a continuación: **no los mueva, modifique ni elimine**.
# 
# Puedes encontrar mis comentarios en cuadros verdes, amarillos o rojos como este:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Exito. Todo se ha hecho de forma exitosa.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Observación. Algunas recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Necesita arreglos. Este apartado necesita algunas correcciones. El trabajo no puede ser aceptado con comentarios rojos. 
# </div>
# 
# Puedes responder utilizando esto:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta de estudiante.</b> <a class="tocSkip"></a>
# </div>

# # Descripción del proyecto
# Trabajas en la compañía de extracción de petróleo OilyGiant. Tu tarea es encontrar los mejores lugares donde abrir 200 pozos nuevos de petróleo.
# Para completar esta tarea, tendrás que realizar los siguientes pasos:
# 
# •	Leer los archivos con los parámetros recogidos de pozos petrolíferos en la región seleccionada: calidad de crudo y volumen de reservas.
# 
# •	Crear un modelo para predecir el volumen de reservas en pozos nuevos.
# 
# •	Elegir los pozos petrolíferos que tienen los valores estimados más altos.
# 
# •	Elegir la región con el beneficio total más alto para los pozos petrolíferos seleccionados.
# 
# Tienes datos sobre muestras de crudo de tres regiones. Ya se conocen los parámetros de cada pozo petrolero de la región.
# Crea un modelo que ayude a elegir la región con el mayor margen de beneficio. 
# Analiza los beneficios y riesgos potenciales utilizando la técnica bootstrapping.
# 
# Condiciones:
# 
# •	Solo se debe usar la regresión lineal para el entrenamiento del modelo.
# 
# •	Al explorar la región, se lleva a cabo un estudio de 500 puntos con la selección de los mejores 200 puntos para el cálculo del beneficio.
# 
# •	El presupuesto para el desarrollo de 200 pozos petroleros es de 100 millones de dólares.
# 
# •	Un barril de materias primas genera 4.5 USD de ingresos. El ingreso de una unidad de producto es de 4500 dólares (el volumen de reservas está expresado en miles de barriles).
# 
# •	Después de la evaluación de riesgo, mantén solo las regiones con riesgo de pérdidas inferior al 2.5%. De las que se ajustan a los criterios, se debe seleccionar la región con el beneficio promedio más alto.
# 
# Los datos son sintéticos: los detalles del contrato y las características del pozo no se publican.
# 
# Descripción de datos
# Los datos de exploración geológica de las tres regiones se almacenan en archivos:
# - geo_data_0.csv. Descarga el conjunto de datos
# - geo_data_1.csv. Descarga el conjunto de datos
# - geo_data_2.csv. Descarga el conjunto de datos
# - id — identificador único de pozo de petróleo
# - f0, f1, f2 — tres características de los puntos (su significado específico no es importante, pero las características en sí son significativas)
# - product — volumen de reservas en el pozo de petróleo (miles de barriles).
# 
# Instrucciones del proyecto:
# 
# ## Paso 1.	Descarga y prepara los datos. Explica el procedimiento.
# ## Paso 2.	Entrena y prueba el modelo para cada región en geo_data_0.csv:
# - 2.1.	Divide los datos en un conjunto de entrenamiento y un conjunto de validación en una proporción de 75:25
# - 2.2	Entrena el modelo y haz predicciones para el conjunto de validación.
# - 2.3	Guarda las predicciones y las respuestas correctas para el conjunto de validación.
# - 2.4	Muestra el volumen medio de reservas predicho y RMSE del modelo.
# - 2.5	Analiza los resultados. Coloca todos los pasos previos en funciones, realiza y ejecuta los pasos 2.1-2.5 para los archivos 'geo_data_1.csv' y 'geo_data_2.csv'.
# 
# ## Paso 3.	Prepárate para el cálculo de ganancias:
# - 3.1	Almacena todos los valores necesarios para los cálculos en variables separadas.
# - 3.2	Dada la inversión de 100 millones por 200 pozos petrolíferos, de media un pozo petrolífero debe producir al menos un valor de 500,000 dólares en unidades para evitar pérdidas (esto es equivalente a 111.1 unidades). Compara esta cantidad con la cantidad media de reservas en cada región.
# - 3.3	Presenta conclusiones sobre cómo preparar el paso para calcular el beneficio.
# 
# ## Paso 4.	Escribe una función para calcular la ganancia de un conjunto de pozos de petróleo seleccionados y modela las predicciones:
# - 4.1	Elige los 200 pozos con los valores de predicción más altos de cada una de las 3 regiones (es decir, archivos 'csv').
# - 4.2	Resume el volumen objetivo de reservas según dichas predicciones. Almacena las predicciones para los 200 pozos para cada una de las 3 regiones.
# - 4.3	Calcula la ganancia potencial de los 200 pozos principales por región. 
# - 4.4   Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección.
# 
# ## Paso 5.	Calcula riesgos y ganancias para cada región:
# - 5.1	Utilizando las predicciones que almacenaste en el paso 4.2, emplea la técnica del bootstrapping con 1000 muestras para hallar la distribución de los beneficios.
# - 5.2	Encuentra el beneficio promedio, el intervalo de confianza del 95% y el riesgo de pérdidas. La pérdida es una ganancia negativa, calcúlala como una probabilidad y luego exprésala como un porcentaje.
# - 5.3	Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección. ¿Coincide tu elección con la elección anterior en el punto 4.3?
# 
# 

#  <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 2) </b> <a class="tocSkip"></a>
# 
# Felicitaciones Pia, has implementado los cambios que te he indicado, bien hecho! Hemos alcanzado los objetivos del proyecto por lo que está en condiciones de ser aprobado!
#     
# Felicitaciones y éxitos en tu camino dentro del mundo de los datos!
#     
# Saludos Pia!

# <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 1) </b> <a class="tocSkip"></a>
# 
# 
# Pia, siempre me tomo este tiempo al inicio de tu proyecto para comentar mis apreciaciones generales de esta iteración de tu entrega. 
#     
# 
# Me gusta comenzar dando la bienvenida al mundo de los datos a los estudiantes, te deseo lo mejor y espero que consigas lograr tus objetivos. Personalmente me gusta brindar el siguiente consejo, "Está bien equivocarse, es normal y es lo mejor que te puede pasar. Aprendemos de los errores y eso te hará mejor programadora ya que podrás descubrir cosas a medida que avances y son estas cosas las que te darán esa experiencia para ser una gran Data Scientist"
#     
# Ahora si yendo a esta notebook. Quería felicitarte Pia porque has realizado de forma excelente el trabajo, has implementado de forma perfecta cada apartado y con un gran compromiso a la hora de mostrar los resultados parciales. Solo hemos tenido un pequeño detalle pero que estoy seguro que te demorará unos segundos corregir.
#     
# Espero con ansias a nuestra próxima iteración Pia, exitos y saludos!

# Paso 1: Descarga y preparación de los datos

# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[21]:


# Cargar los datos
geo_data_0 = pd.read_csv('/datasets/geo_data_0.csv')
geo_data_1 = pd.read_csv('/datasets/geo_data_1.csv')
geo_data_2 = pd.read_csv('/datasets/geo_data_2.csv')

# Mostrar las primeras filas de cada conjunto de datos
print(geo_data_0.head())
print(geo_data_1.head())
print(geo_data_2.head())


# In[22]:


# Información general y descripción de los datos
print(geo_data_0.info())

print(geo_data_1.info())

print(geo_data_2.info())

# Comprobación de valores nulos
print(geo_data_0.isnull().sum())
print(geo_data_1.isnull().sum())
print(geo_data_2.isnull().sum())


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Excelente carga de los datos manteniendolos por separados de las importaciones de librerías como así la implementación de métodos para profundizar en la comprensión de nuestros datos. Bien hecho!

# Paso 2: Entrenamiento y Prueba del Modelo
# Vamos a crear un modelo de regresión lineal para predecir el volumen de reservas en pozos nuevos.
# 
# 2.1 División de los Datos
# Dividimos los datos en conjuntos de entrenamiento y validación en una proporción de 75:25.

# In[23]:


# Dividir los datos
def split_data(data):
    X = data.drop(['id', 'product'], axis=1)
    y = data['product']
    return train_test_split(X, y, test_size=0.25, random_state=42)

X_train_0, X_val_0, y_train_0, y_val_0 = split_data(geo_data_0)
X_train_1, X_val_1, y_train_1, y_val_1 = split_data(geo_data_1)
X_train_2, X_val_2, y_train_2, y_val_2 = split_data(geo_data_2)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Perfecta cración de la función a cargo de la división de los datos, a la vez bien tenido en cuenta el quitar aquella feautre que no aporta valor como los id.

# 2.2 Entrenamiento del Modelo
# Entrenamos un modelo de regresión lineal con los datos de entrenamiento y hacemos predicciones sobre el conjunto de validación.

# In[24]:


# Función para entrenar y evaluar el modelo
def train_and_evaluate(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return predictions, rmse

# Entrenar y evaluar modelos para cada región
predictions_0, rmse_0 = train_and_evaluate(X_train_0, y_train_0, X_val_0, y_val_0)
predictions_1, rmse_1 = train_and_evaluate(X_train_1, y_train_1, X_val_1, y_val_1)
predictions_2, rmse_2 = train_and_evaluate(X_train_2, y_train_2, X_val_2, y_val_2)

# Resultados
print("Región 0 - RMSE:", rmse_0)
print("Región 1 - RMSE:", rmse_1)
print("Región 2 - RMSE:", rmse_2)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Nuevamente felicitaciones, esta vez por crear la función a cargo del entrenamiento. También una excelente implementación, llegamos a los resultados esperados!

# 2.3 Guardar Predicciones y Respuestas Correctas
# Guardamos las predicciones y las respuestas correctas para el conjunto de validación.

# In[25]:


# Guardar predicciones y respuestas correctas
validation_results_0 = pd.DataFrame({'actual': y_val_0, 'predicted': predictions_0})
validation_results_1 = pd.DataFrame({'actual': y_val_1, 'predicted': predictions_1})
validation_results_2 = pd.DataFrame({'actual': y_val_2, 'predicted': predictions_2})

# Mostrar algunas filas de los resultados
print(validation_results_0.head())
print(validation_results_1.head())
print(validation_results_2.head())


# 2.4 Volumen Medio de Reservas Predicho y RMSE del Modelo
# Calculamos y mostramos el volumen medio de reservas predicho y el RMSE del modelo para cada región.

# In[26]:


# Volumen medio de reservas predicho
mean_predictions_0 = predictions_0.mean()
mean_predictions_1 = predictions_1.mean()
mean_predictions_2 = predictions_2.mean()

# Resultados
print("Región 0 - Volumen medio de reservas predicho:", mean_predictions_0)
print("Región 1 - Volumen medio de reservas predicho:", mean_predictions_1)
print("Región 2 - Volumen medio de reservas predicho:", mean_predictions_2)


# 2.5 Análisis de Resultados
# Analizamos los resultados obtenidos de las predicciones y los errores.

# In[27]:


# Análisis de resultados
def analyze_results(mean_predictions, rmse, region):
    print(f"Región {region}:")
    print(f" - Volumen medio de reservas predicho: {mean_predictions}")
    print(f" - RMSE del modelo: {rmse}")

analyze_results(mean_predictions_0, rmse_0, 0)
analyze_results(mean_predictions_1, rmse_1, 1)
analyze_results(mean_predictions_2, rmse_2, 2)


# Paso 3: Preparación para el Cálculo de Ganancias

# 3.1 Almacenar valores necesarios para los cálculos

# In[28]:


# Almacenar valores necesarios para los cálculos
BUDGET = 100e6  # Presupuesto total en dólares
WELLS = 200  # Número de pozos
REVENUE_PER_UNIT = 4500  # Ingreso por unidad (1000 barriles)
MIN_REVENUE_PER_WELL = BUDGET / WELLS  # Ingreso mínimo por pozo
MIN_UNITS_PER_WELL = MIN_REVENUE_PER_WELL / REVENUE_PER_UNIT  # Unidades mínimas por pozo

print("Cantidad mínima requerida por pozo:", MIN_UNITS_PER_WELL)


# 3.2 Comparar con la cantidad media de reservas en cada región

# In[29]:


# Comparación con cantidad media de reservas
mean_reserves_0 = geo_data_0['product'].mean()
mean_reserves_1 = geo_data_1['product'].mean()
mean_reserves_2 = geo_data_2['product'].mean()

print("Región 0 - Cantidad media de reservas:", mean_reserves_0)
print("Región 1 - Cantidad media de reservas:", mean_reserves_1)
print("Región 2 - Cantidad media de reservas:", mean_reserves_2)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Bien hecho! Excelentes promedios!

# 3.3 Conclusiones sobre cómo preparar el paso para calcular el beneficio
# Para calcular el beneficio, debemos asegurarnos de que los pozos seleccionados cumplan con la cantidad mínima de reservas requerida para evitar pérdidas. Procederemos a seleccionar los pozos con las mayores reservas predichas y calcular el beneficio potencial.

# Paso 4: Cálculo de Ganancias

# 4.1 Elige los 200 pozos con los valores de predicción más altos

# In[30]:


# Función para calcular la ganancia
def calculate_profit(predictions, target, count, revenue_per_unit):
    sorted_indices = predictions.argsort()[::-1][:count]
    selected_reserves = target.iloc[sorted_indices]
    total_revenue = selected_reserves.sum() * revenue_per_unit
    return total_revenue - BUDGET

# Calcular ganancias para cada región
profit_0 = calculate_profit(predictions_0, y_val_0, WELLS, REVENUE_PER_UNIT)
profit_1 = calculate_profit(predictions_1, y_val_1, WELLS, REVENUE_PER_UNIT)
profit_2 = calculate_profit(predictions_2, y_val_2, WELLS, REVENUE_PER_UNIT)

print("Región 0 - Ganancia potencial:", profit_0)
print("Región 1 - Ganancia potencial:", profit_1)
print("Región 2 - Ganancia potencial:", profit_2)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Perfecto Pia! Tal como se debía hemos tomado los top 200 pozos en base a las predicciones y de forma correcta hemos calculado la ganancia al multiplicar por la ganancia por unidad y al restar el presupuesto. Sigamos!

# 4.2 Resume el volumen objetivo de reservas y almacena las predicciones

# In[31]:


# Convertir las predicciones en Series para usar índices
predictions_0_series = pd.Series(predictions_0)
predictions_1_series = pd.Series(predictions_1)
predictions_2_series = pd.Series(predictions_2)

# Almacenar predicciones para los 200 pozos seleccionados
top_200_predictions_0 = predictions_0_series.nlargest(200)
top_200_predictions_1 = predictions_1_series.nlargest(200)
top_200_predictions_2 = predictions_2_series.nlargest(200)

# Volumen objetivo de reservas
top_200_reserves_0 = y_val_0.iloc[top_200_predictions_0.index].sum()
top_200_reserves_1 = y_val_1.iloc[top_200_predictions_1.index].sum()
top_200_reserves_2 = y_val_2.iloc[top_200_predictions_2.index].sum()

print("Región 0 - Volumen objetivo de reservas (200 pozos):", top_200_reserves_0)
print("Región 1 - Volumen objetivo de reservas (200 pozos):", top_200_reserves_1)
print("Región 2 - Volumen objetivo de reservas (200 pozos):", top_200_reserves_2)


# 4.3 Calcula la ganancia potencial de los 200 pozos principales por región
# Calculado previamente en el paso 4.1.

# 4.4 Presenta tus conclusiones y justifica tu elección
# Basado en las ganancias potenciales calculadas, seleccionamos la región con la ganancia más alta y justificamos nuestra elección.

# In[32]:


# Conclusiones
def present_conclusions(profits, reserves, region):
    print(f"Región {region}:")
    print(f" - Ganancia potencial: {profits}")
    print(f" - Volumen objetivo de reservas: {reserves}")

present_conclusions(profit_0, top_200_reserves_0, 0)
present_conclusions(profit_1, top_200_reserves_1, 1)
present_conclusions(profit_2, top_200_reserves_2, 2)

best_region = max([(0, profit_0), (1, profit_1), (2, profit_2)], key=lambda x: x[1])[0]
print(f"La mejor región para el desarrollo de pozos es la región {best_region} debido a su mayor ganancia potencial.")


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Excelentes agregado al calcular tanto potenciales ganancias como volumnes objetivos por region! Impresionante Pia!

# Paso 5: Cálculo de Riesgos y Ganancias Utilizando Bootstrapping

# 5.1 Utiliza la técnica de bootstrapping con 1000 muestras

# In[36]:


# Función de bootstrapping
def bootstrap_profit(predictions, target, count, revenue_per_unit, n_samples=1000):
    profits = []
    for _ in range(n_samples):
        sample_indices = np.random.choice(predictions.index, size=500, replace=True)
        sample_predictions = predictions[sample_indices]
        sample_target = target.iloc[sample_indices]
        profit = calculate_profit(sample_predictions, sample_target, count, revenue_per_unit)
        profits.append(profit)
    return np.array(profits)

# Realizar bootstrapping para cada región
profits_0 = bootstrap_profit(pd.Series(predictions_0), y_val_0, WELLS, REVENUE_PER_UNIT)
profits_1 = bootstrap_profit(pd.Series(predictions_1), y_val_1, WELLS, REVENUE_PER_UNIT)
profits_2 = bootstrap_profit(pd.Series(predictions_2), y_val_2, WELLS, REVENUE_PER_UNIT)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Aquí lo hemos hehco muy bien Pia, haz en su mayoría implementado las cosas que debíamos sin embargo aún nos faltan corregir algunos aspectos. De forma correcta estamos iterando 1000 veces y calculando las ganancias sobre la muestre como a la vez luego guardandola para calcular los percentiles inferiores y superios como así el riesgo de perdida en la siguiente función. Sin embargo, nos ha faltado un detalle y este refiere a que en cada iteración estamos tomando la totalidad de los datos cuando en realidad lo que debemos hacer es tomar una muestra de datos más chica, por ejemplo 500 observaciones de muestra en cada una de las 1000 iteraciones, actualmente estamos tomando size=len(predictions) que refiere a la totalidad de los datos.
#     
#     Excelente implementación de lo indicado!

# <b>Respuesta de estudiante.</b> <a class="tocSkip"></a>
# </div>
# Corregido 
# 

# 5.2 Encuentra el beneficio promedio, el intervalo de confianza del 95% y el riesgo de pérdidas

# In[37]:


# Análisis de resultados de bootstrapping
def analyze_bootstrap_results(profits):
    mean_profit = np.mean(profits)
    lower_bound = np.percentile(profits, 2.5)
    upper_bound = np.percentile(profits, 97.5)
    risk_of_loss = (profits < 0).mean()
    return mean_profit, lower_bound, upper_bound, risk_of_loss

mean_profit_0, lower_bound_0, upper_bound_0, risk_of_loss_0 = analyze_bootstrap_results(profits_0)
mean_profit_1, lower_bound_1, upper_bound_1, risk_of_loss_1 = analyze_bootstrap_results(profits_1)
mean_profit_2, lower_bound_2, upper_bound_2, risk_of_loss_2 = analyze_bootstrap_results(profits_2)

print("Región 0 - Beneficio promedio:", mean_profit_0)
print("Región 0 - Intervalo de confianza 95%:", (lower_bound_0, upper_bound_0))
print("Región 0 - Riesgo de pérdidas:", risk_of_loss_0)

print("Región 1 - Beneficio promedio:", mean_profit_1)
print("Región 1 - Intervalo de confianza 95%:", (lower_bound_1, upper_bound_1))
print("Región 1 - Riesgo de pérdidas:", risk_of_loss_1)

print("Región 2 - Beneficio promedio:", mean_profit_2)
print("Región 2 - Intervalo de confianza 95%:", (lower_bound_2, upper_bound_2))
print("Región 2 - Riesgo de pérdidas:", risk_of_loss_2)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Aquí lo hemos hecho muy bien, hemos de forma correcta tomado los profist provenientes del bootstrapping como así hemos calculados los percentiles inferiores y superiores . Simplemente deberíamos ver unos pocos cambios una vez corrijamos lo indicado anteriormente.

# <b>Respuesta de estudiante.</b> <a class="tocSkip"></a>
# </div>
# Resultado con cambios dado a la corrección.

# 5.3 Presenta tus conclusiones y justifica tu elección
# Con base en el análisis de riesgos y ganancias, proponemos la mejor región para el desarrollo de pozos.

# In[35]:


# Conclusiones finales
def present_final_conclusions(mean_profit, lower_bound, upper_bound, risk_of_loss, region):
    print(f"Región {region}:")
    print(f" - Beneficio promedio: {mean_profit}")
    print(f" - Intervalo de confianza 95%: ({lower_bound}, {upper_bound})")
    print(f" - Riesgo de pérdidas: {risk_of_loss}")

present_final_conclusions(mean_profit_0, lower_bound_0, upper_bound_0, risk_of_loss_0, 0)
present_final_conclusions(mean_profit_1, lower_bound_1, upper_bound_1, risk_of_loss_1, 1)
present_final_conclusions(mean_profit_2, lower_bound_2, upper_bound_2, risk_of_loss_2, 2)

best_region = max([(0, mean_profit_0), (1, mean_profit_1), (2, mean_profit_2)], key=lambda x: x[1])[0]
print(f"La mejor región para el desarrollo de pozos es la región {best_region} debido a su mayor ganancia potencial.")


# <b>Respuesta de estudiante.</b> <a class="tocSkip"></a>
# </div>
# Resultado con cambios dado a la corrección.

# 1. ¿Cuáles son tus hallazgos sobre el estudio de tareas?
# Región 0:
# Tiene el volumen medio de reservas predicho más alto (92.3988) y una RMSE de 37.7566, lo que indica una buena precisión del modelo.
# Región 1:
# Tiene el volumen medio de reservas predicho más bajo (68.7129) y una RMSE extremadamente baja de 0.8903, lo que sugiere que el modelo puede estar sobreajustado o que los datos son muy consistentes.
# Región 2:
# Tiene un volumen medio de reservas predicho (94.7710) similar al de la Región 0, pero con una RMSE de 40.1459, lo que sugiere una mayor variabilidad en los datos de esta región.
# 
# 2. ¿Sugeriste la mejor región para el desarrollo de pozos? ¿Justificaste tu elección?
# 
# Mejor región para el desarrollo de pozos: La Región 1.
# 
# La Región 1 tiene el beneficio promedio más alto después del ajuste utilizando la técnica de bootstrapping correctamente.
# El análisis de bootstrapping mostró que la Región 1 tiene un beneficio promedio de 4,455,171.62 dólares y un intervalo de confianza del 95% que garantiza una mayor estabilidad en los beneficios.
# Además, la Región 1 tiene el riesgo de pérdidas más bajo (1.8%), lo que refuerza la elección debido a la menor incertidumbre asociada.
# 
# Resumen de los resultados:
# 
# Región 0:
# 
# Beneficio promedio: 3,946,406.20 dólares
# Intervalo de confianza 95%: (-1,651,365.00, 9,019,799.16)
# Riesgo de pérdidas: 7%
# 
# Región 1:
# 
# Beneficio promedio: 4,455,171.62 dólares
# Intervalo de confianza 95%: (553,121.07, 8,438,567.11)
# Riesgo de pérdidas: 1.8%
# 
# Región 2:
# 
# Beneficio promedio: 3,751,628.65 dólares
# Intervalo de confianza 95%: (-1,306,189.96, 9,061,252.22)
# Riesgo de pérdidas: 7.5%
# 
# * Conclusión:
# 
# La Región 1 es la mejor opción para el desarrollo de pozos debido a su mayor ganancia potencial promedio y su menor riesgo de pérdidas. La justificación se basa en los siguientes puntos:
# 
# Beneficio promedio más alto: 4,455,171.62 dólares.
# Intervalo de confianza del 95% que muestra estabilidad en los beneficios: (553,121.07, 8,438,567.11).
# Riesgo de pérdidas más bajo: 1.8%.
# Este cambio de conclusión se basa en un análisis más preciso y ajustado utilizando la técnica de bootstrapping correctamente, asegurando que se tomen muestras más representativas y estables.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Quería dedicarme aquí a agradecerte Pia por que has hecho un trabajo de exelencia, sobre todo con tus puestas en comun y conclusiones, desde las implementaciones detalladas y los agregados a lo solicitado en el proyecto. Solo hemos tenidos uno detalle pero que podremos resolver sin lugar a duda.

# In[ ]:





# In[ ]:




