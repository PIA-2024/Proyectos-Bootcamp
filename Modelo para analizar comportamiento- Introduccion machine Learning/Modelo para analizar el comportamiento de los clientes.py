# Paso 1: Cargar y examinar datos

import pandas as pd
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv('C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS BOOTCAMP/Modelo para analizar comportamiento- Introduccion machine Learning/users_behavior.csv')

# Mostrar las primeras filas del dataset y la información general
data.head(), data.info(), data.describe()

# # Paso 2: Segmentación de los Datos

from sklearn.model_selection import train_test_split

# Primero, dividimos los datos en entrenamiento y temporal (combinando validación y prueba)
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=123)

# Luego, dividimos los datos temporales en validación y prueba
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=123)

# Imprimir las dimensiones de cada conjunto
print("Tamaño del conjunto de entrenamiento:", train_data.shape)
print("Tamaño del conjunto de validación:", validation_data.shape)
print("Tamaño del conjunto de prueba:", test_data.shape)

# # Paso 3: Investigación de Modelos

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

print(results)

# # Paso 4: Ajuste de Hiperparámetros

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

# # Paso 5: Evaluación Final en el Conjunto de Prueba

from sklearn.metrics import accuracy_score

# Preparar las características y la variable objetivo del conjunto de prueba
features_test = test_data.drop(columns=['is_ultra'])
target_test = test_data['is_ultra']

# Utilizar el mejor modelo para hacer predicciones en el conjunto de prueba
test_predictions = best_model.predict(features_test)

# Calcular y mostrar la exactitud en el conjunto de prueba
test_accuracy = accuracy_score(target_test, test_predictions)
print("Exactitud en el conjunto de prueba:", test_accuracy)

# Prueba de Cordura (Paso 5 Adicional)


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


importances = best_model.feature_importances_
features = ['calls', 'minutes', 'messages', 'mb_used']
importance_dict = dict(zip(features, importances))

print("Importancia de las características:", importance_dict)

# Predicciones para Datos Extremos:

# El modelo predice "1" (plan Ultra) para ambos casos extremos (mínimos y máximos en uso de llamadas, minutos, mensajes y datos). Esto sugiere que el modelo podría estar sesgado hacia la clasificación de usuarios en el plan Ultra cuando se presentan valores extremos. Este comportamiento puede ser un área de interés para investigar más a fondo, especialmente en términos de cómo el balance de clases y la distribución de las características afectan el entrenamiento del modelo.

# Importancia de las Características:

# mb_used: 33.54% - Esta es la característica más influyente según el modelo, lo cual tiene sentido intuitivamente, ya que el uso de datos podría ser un buen indicador de la necesidad de un plan más robusto como el Ultra.
# minutes: 22.98% - La duración de las llamadas también es significativa, lo que es razonable dado que podría correlacionarse con un uso más intensivo del plan.
# calls: 21.87% - Similar a minutes, el número de llamadas también influye notablemente en la clasificación.
# messages: 21.61% - Aunque también es importante, tiene un poco menos de peso que las llamadas y los minutos, lo que podría reflejar una menor variabilidad en el número de mensajes entre los usuarios de diferentes planes o una menor relevancia de los mensajes en la decisión del plan.

