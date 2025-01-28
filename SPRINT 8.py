#!/usr/bin/env python
# coding: utf-8

# ¡Hola Pia! Como te va?
# 
# Mi nombre es Facundo Lozano! Un gusto conocerte, seré tu revisor en este proyecto.
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
# </div> Hola, Facundo, muchas gracias por tus comentarios en mi proyecto, ayuda mucho a ir aprendiendo y perfeccionando las habilidades aprendidas en TripleTen. Agradecida con tus buenas vibras para continuar aprendidendo y teniendo éxitos en mis estudios. Saludos.

# # Descripción del proyecto
# La compañía móvil Megaline no está satisfecha al ver que muchos de sus clientes utilizan planes heredados. Quieren desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart o Ultra.
# Tienes acceso a los datos de comportamiento de los suscriptores que ya se han cambiado a los planes nuevos (del proyecto del sprint de Análisis estadístico de datos). Para esta tarea de clasificación debes crear un modelo que escoja el plan correcto. Como ya hiciste el paso de procesar los datos, puedes lanzarte directo a crear el modelo.
# Desarrolla un modelo con la mayor exactitud posible. En este proyecto, el umbral de exactitud es 0.75. Usa el dataset para comprobar la exactitud.
# Instrucciones del proyecto.
# ## 	Abre y examina el archivo de datos. Dirección al archivo:datasets/users_behavior.csv Descarga el dataset
# ## 	Segmenta los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.
# ## 	Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio.
# ## 	Comprueba la calidad del modelo usando el conjunto de prueba.
# ## 	Tarea adicional: haz una prueba de cordura al modelo. Estos datos son más complejos que los que habías usado antes así que no será una tarea fácil. Más adelante lo veremos con más detalle.
# # Descripción de datos
# Cada observación en el dataset contiene información del comportamiento mensual sobre un usuario. La información dada es la siguiente:
# •	сalls — número de llamadas,
# •	minutes — duración total de la llamada en minutos,
# •	messages — número de mensajes de texto,
# •	mb_used — Tráfico de Internet utilizado en MB,
# •	is_ultra — plan para el mes actual (Ultra - 1, Smart - 0).

# <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 1) </b> <a class="tocSkip"></a>
# 
# Buenas Pia! Siempre me tomo este tiempo al inicio de tu proyecto para comentarte mis apreciaciones generales de esta iteración de tu entrega. 
# 
# Me gusta comenzar dando la bienvenida al mundo de los datos a los estudiantes, te deseo lo mejor y espero que consigas lograr tus objetivos. Personalmente me gusta brindar el siguiente consejo, "Está bien equivocarse, es normal y es lo mejor que te puede pasar. Aprendemos de los errores y eso te hará mejor programando ya que podrás descubrir cosas a medida que avances y son estas cosas las que te darán esa experiencia para ser mejor como  Data Scientist"
# 
# Ahora si yendo a esta notebook. Lo he dicho al final del proyecto pero lo resalto aquí nuevamente, tu proyecto está muy bien resuelto Pia, resalta capacidad y comprensión de todas las herrramientas, como a la vez esta ordenado y es sencillo de seguir, felictiaciones!
# 
# Este proyecto está en condiciones de ser aprobado! Éxitos dentro de tu camino en el mundo de los datos!
# 
# Saludos Pia!

# # Paso 1: Cargar y examinar datos

# In[1]:


import pandas as pd


# In[2]:


# Cargar los datos desde el archivo CSV
data = pd.read_csv('/datasets/users_behavior.csv')

# Mostrar las primeras filas del dataset y la información general
data.head(), data.info(), data.describe()


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Excelente implementación de importaciones y carga de datos. Tengamos en cuenta que siempre es más positivo mantener los procesos en celdas separadas! A la vez excelente implementación de los métodos para observar la composición de los datos!

# # Paso 2: Segmentación de los Datos

# In[3]:


from sklearn.model_selection import train_test_split

# Primero, dividimos los datos en entrenamiento y temporal (combinando validación y prueba)
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=123)

# Luego, dividimos los datos temporales en validación y prueba
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=123)

# Imprimir las dimensiones de cada conjunto
print("Tamaño del conjunto de entrenamiento:", train_data.shape)
print("Tamaño del conjunto de validación:", validation_data.shape)
print("Tamaño del conjunto de prueba:", test_data.shape)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Excelente división de los datos en los 3 conjuntos correpsondientes Pia, excelente implementación de train_test_split()!

# # Paso 3: Investigación de Modelos

# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Preparar los conjuntos de datos
features_train = train_data.drop(columns=['is_ultra'])
target_train = train_data['is_ultra']
features_validation = validation_data.drop(columns=['is_ultra'])
target_validation = validation_data['is_ultra']

# Modelos a evaluar
models = {
    "Logistic Regression": LogisticRegression(random_state=123, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=123),
    "Random Forest": RandomForestClassifier(random_state=123)
}

# Entrenar y evaluar cada modelo
results = {}
for name, model in models.items():
    model.fit(features_train, target_train)
    predictions = model.predict(features_validation)
    accuracy = accuracy_score(target_validation, predictions)
    results[name] = accuracy

results


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Procedimiento perfecto Pia! En este caso guardando diferentes modelos basicos para luego iterar, entrenarlos y evaluarlos con las metricas correspondientes, muy bien hecho!

# # Paso 4: Ajuste de Hiperparámetros

# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}

# Inicializar el modelo GridSearchCV con el modelo RandomForest y la cuadrícula de parámetros
grid_search = GridSearchCV(RandomForestClassifier(random_state=123), param_grid, cv=3, scoring='accuracy')

# Entrenar GridSearchCV en los datos de entrenamiento
grid_search.fit(features_train, target_train)

# Obtener el mejor modelo y su puntuación
best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

# Imprimir los mejores hiperparámetros y la mejor exactitud
print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor exactitud de cross-validation:", best_accuracy)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Y tal como era debido una excelente profundización implementando una alteantiva como GridSearch para ver el impacto de diferenes hiperparametros y por ende obtener el mejor modelo posible.

# # Paso 5: Evaluación Final en el Conjunto de Prueba

# In[6]:


from sklearn.metrics import accuracy_score

# Preparar las características y la variable objetivo del conjunto de prueba
features_test = test_data.drop(columns=['is_ultra'])
target_test = test_data['is_ultra']

# Utilizar el mejor modelo para hacer predicciones en el conjunto de prueba
test_predictions = best_model.predict(features_test)

# Calcular y mostrar la exactitud en el conjunto de prueba
test_accuracy = accuracy_score(target_test, test_predictions)
print("Exactitud en el conjunto de prueba:", test_accuracy)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Y una excelente prueba del mejor modelo contra el tercer conjunto, bien hecho!

# # Prueba de Cordura (Paso 5 Adicional)

# In[7]:


import numpy as np

# Crear datos de entrada extremos
extreme_data = pd.DataFrame({
    'calls': [0, 300],  # Muy pocas llamadas y muchas llamadas
    'minutes': [0, 2000],  # Ninguna duración y duración muy larga
    'messages': [0, 500],  # Ningún mensaje y muchos mensajes
    'mb_used': [0, 50000]  # Sin uso de datos y uso extremadamente alto
})

# Hacer predicciones con el mejor modelo
extreme_predictions = best_model.predict(extreme_data)
print("Predicciones para datos extremos:", extreme_predictions)


# In[8]:


importances = best_model.feature_importances_
features = ['calls', 'minutes', 'messages', 'mb_used']
importance_dict = dict(zip(features, importances))

print("Importancia de las características:", importance_dict)


# <div class="alert alert-block alert-warning">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Cuidadoa quí Pia, antes que nada al ser una sección opcional lo marcaré como una recomendación. La intención del modelo de cordura es perfecta pero debemos tener en cuenta que un modelo de cordura representa una un modelo muy básico que debemos utilizar contra nuestro conjunto de prueba, la intención de este modelo es asegurarnos que el modelo creado por nosotros tiene mejores resultados que este básico (de ahí el nombre de cordura) asegurandonos así que vale la pena crear un modelo de machine learning. En este caso, estamos utilizando directamente el modelo que ya habíamos creados por lo que no cumple con ser un modelo básico, te recomiendo investigar e implementar modelos como DummyClassifier().

# Predicciones para Datos Extremos:
# 
# El modelo predice "1" (plan Ultra) para ambos casos extremos (mínimos y máximos en uso de llamadas, minutos, mensajes y datos). Esto sugiere que el modelo podría estar sesgado hacia la clasificación de usuarios en el plan Ultra cuando se presentan valores extremos. Este comportamiento puede ser un área de interés para investigar más a fondo, especialmente en términos de cómo el balance de clases y la distribución de las características afectan el entrenamiento del modelo.
# 
# Importancia de las Características:
# 
# mb_used: 33.54% - Esta es la característica más influyente según el modelo, lo cual tiene sentido intuitivamente, ya que el uso de datos podría ser un buen indicador de la necesidad de un plan más robusto como el Ultra.
# minutes: 22.98% - La duración de las llamadas también es significativa, lo que es razonable dado que podría correlacionarse con un uso más intensivo del plan.
# calls: 21.87% - Similar a minutes, el número de llamadas también influye notablemente en la clasificación.
# messages: 21.61% - Aunque también es importante, tiene un poco menos de peso que las llamadas y los minutos, lo que podría reflejar una menor variabilidad en el número de mensajes entre los usuarios de diferentes planes o una menor relevancia de los mensajes en la decisión del plan.

# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Excelente conclusión de lo trabajado Pia, denota una gran comprensión de lo implementado y lo obtenido, muy bien hecho!

# In[ ]:




