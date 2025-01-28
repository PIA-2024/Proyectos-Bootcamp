#!/usr/bin/env python
# coding: utf-8

# Mi nombre es Pia! Un gusto conocerte, seré tu revisor en este proyecto.
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

# # Descripción del proyecto:
# Los clientes de Beta Bank se están yendo, cada mes, poco a poco. Los banqueros descubrieron que es más barato salvar a los clientes existentes que atraer nuevos.
# 
# Necesitamos predecir si un cliente dejará el banco pronto. Tú tienes los datos sobre el comportamiento pasado de los clientes y la terminación de contratos con el banco.
# 
# -Crea un modelo con el máximo valor F1 posible. Para aprobar la revisión, necesitas un valor F1 de al menos 0.59. Verifica F1 para el conjunto de prueba. 
# 
# -Además, debes medir la métrica AUC-ROC y compararla con el valor F1.
# 
# -Descarga y prepara los datos.  Explica el procedimiento.
# 
# -Examina el equilibrio de clases. Entrena el modelo sin tener en cuenta el desequilibrio. Describe brevemente tus hallazgos.
# 
# -Mejora la calidad del modelo. Asegúrate de utilizar al menos dos enfoques para corregir el desequilibrio de clases. Utiliza conjuntos de entrenamiento y validación para encontrar el mejor modelo y el mejor conjunto de parámetros. Entrena diferentes modelos en los conjuntos de entrenamiento y validación. Encuentra el mejor. Describe brevemente tus hallazgos.
# 
# -Realiza la prueba final.
# 
# Objetivo
# 
# Exited: El cliente se ha ido (1 - sí; 0 - no)

# <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 1) </b> <a class="tocSkip"></a>
# 
# Hola Pia, quería tomarme este tiempo al inicio de tu proyecto para comentarte mis apreciaciones generales de esta iteración de tu entrega.
#     
# 
# Siempre me gusta comenzar dando la bienvenida al mundo de los datos a los estudiantes, te deseo lo mejor y espero que consigas lograr tus objetivos. Personalmente me gusta brindar el siguiente consejo, "Está bien equivocarse, es normal y es lo mejor que te puede pasar. Aprendemos de los errores y eso te hará mejor programando ya que podrás descubrir cosas a medida que avances y son estas cosas las que te darán esa experiencia para ser mejor como Data Scientist."
#     
# Ahora si yendo a esta notebook.  Lo he dicho al final del proyecto pero lo resalto aquí nuevamente Pia, tu proyecto está muy bien resuelto, resalta capacidad y comprensión de todas las herramientas, como a la vez esta ordenado y es sencillo de seguir, felictiaciones!
# 
# Este proyecto está en condiciones de ser aprobado! Éxitos dentro de tu camino en el mundo de los datos!
# 
# Saludos!

# Parte 1: importar librerias, descargar datos y analizar datos

# In[1]:


# importar librerias
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, mean_squared_error, mean_absolute_error, precision_recall_curve, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Cargar los datos
data = pd.read_csv('/datasets/Churn.csv')

# Inspección inicial de los datos
print(data.head())
print(data.describe())
print(data.info())


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Excelente Pia! Hasta aquí hemos importado correctamente las librerías manteniendolas por separado de la carga de los datos y tamibién hemos implementado correctamente los métodos para observar la composición de nuestros datos, bien hecho!

# Procesar los datos

# In[3]:


# Eliminar columnas innecesarias
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Codificación One-Hot de variables categóricas
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)


# Escalado de características numéricas
scaler = StandardScaler()
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Mostrar el conjunto de datos preprocesado y los valores faltantes
data.head()


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Y tal como era recomendado has eliminado aquellas features que no aportan valor al modelo, excelente! A la vez una perfecta implementación de transformaciones tanto a nuestras features categóricas como numéricas, bien hecho!

# Los datos han sido preprocesados de la siguiente manera:
# 
# Columnas eliminadas: RowNumber, CustomerId, y Surname.
# Codificación One-Hot: Geography y Gender.
# Escalado de características numéricas: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary.
# Valores faltantes:
# Hay valores faltantes en la columna Tenure (909 valores).
# Para el manejo de valores faltantes en Tenure voy a obtar por rellenar los valores con la mediana para mantener la mayor cantidad de datos posible.

# In[4]:


# Rellenar valores faltantes en 'Tenure' con la mediana
data['Tenure'].fillna(data['Tenure'].median(), inplace=True)

# Verificar nuevamente los valores faltantes
missing_values_after = data.isnull().sum()
print(missing_values_after)
# Mostrar el conjunto de datos preprocesado
print(data.head())


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Perfecto relleno sobre los valores nulos!

# Parte 2: Entrenamiento de Modelos y Evaluación Inicial

# In[5]:


# División de los datos en características y objetivo
X = data.drop('Exited', axis=1)
y = data['Exited']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características con MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo - Uso de Árbol de Decisión para demostrar exactitud en la clasificación
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluación inicial
initial_classification_report = classification_report(y_test, y_pred)
initial_auc_roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
initial_mse = mean_squared_error(y_test, y_pred)
initial_mae = mean_absolute_error(y_test, y_pred)
initial_conf_matrix = confusion_matrix(y_test, y_pred)

print("Reporte de clasificación inicial:")
print(initial_classification_report)
print(f"AUC-ROC inicial: {initial_auc_roc}")
print(f"Error Cuadrático Medio: {initial_mse}")
print(f"Error Absoluto Medio: {initial_mae}")
print(f"Matriz de confusión:\n{initial_conf_matrix}")


# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Cuidado aquí Pia, el procedimiento es perfecto, desde la división de los datos, la creación del modelo basico y el entrenamiento, sin embargo tenemos algunos detalles que podríamos mejorar. Por un lado, estamos escalando nuevamente sobre los datos previamente escalados, si bien no es algo sumamente prohibido te recomendaría quedarte con un solo escalamiento, por otro lado la predicción y cálculo de métricas son correctas pero también hemos agregado el cálculo del error cuadrático medio cuando nuestro problema es un problema de clasificación binaria, tratemos de quitar esto.

# Parte 3: Manejo del Desequilibrio de Clases

# In[6]:


# Examinar el equilibrio de clases
print("Distribución de clases en el objetivo:")
print(y_train.value_counts(normalize=True))

# Ajuste de peso de clase en el modelo
model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
y_pred_balanced = model.predict(X_test_scaled)

# Evaluación del modelo balanceado
balanced_classification_report = classification_report(y_test, y_pred_balanced)
balanced_auc_roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
balanced_mse = mean_squared_error(y_test, y_pred_balanced)
balanced_mae = mean_absolute_error(y_test, y_pred_balanced)

print("Reporte de clasificación con ajuste de peso de clase:")
print(balanced_classification_report)
print(f"AUC-ROC con ajuste de peso de clase: {balanced_auc_roc}")
print(f"Error Cuadrático Medio con ajuste de peso de clase: {balanced_mse}")
print(f"Error Absoluto Medio con ajuste de peso de clase: {balanced_mae}")


# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Excelente observación respecto al desequilibrio de clases Pia, si quisieramos mejorarlo podríamos ver esto mediante una gráfica para que sea más sencillo de observar!

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Nuevamente perfecta implementación del modelo en su bersión básica pero esta vez implementando un método de mitigación del desequilibrio de clases. Bien hecho!

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Recordemos que los errores cuadrados medios son métricas de problemas de regresión, por lo que recomendaría centrarnos en el F1.

# Parte 4: Optimización de Hiperparámetros

# In[7]:


# Optimización de hiperparámetros con GridSearchCV
param_grid = {'max_depth': [10, 20, 30]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train_scaled, y_train)

# Mejor modelo y su rendimiento
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
best_auc_roc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
best_classification_report = classification_report(y_test, y_pred_best)

print("Mejor AUC-ROC:", best_auc_roc)
print("Reporte de clasificación del mejor modelo:")
print(best_classification_report)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Y en este caso perfecta implementación de grilla con GridSearchCV para buscar los mejores hiperparametros posibles. Perfecto Pia!

# Parte 5: Evaluación Avanzada

# In[8]:


# 1. Ajuste de umbral y cálculo de métricas
proba = best_model.predict_proba(X_test_scaled)[:, 1]
thresholds = np.arange(0.0, 1.0, 0.05)
threshold_results = []
for threshold in thresholds:
    y_pred_threshold = (proba >= threshold).astype(int)
    threshold_results.append({
        "threshold": threshold,
        "f1_score": f1_score(y_test, y_pred_threshold),
        "roc_auc": roc_auc_score(y_test, y_pred_threshold),
        "confusion_matrix": confusion_matrix(y_test, y_pred_threshold)
    })

# Mostrar resultados de los umbrales
for result in threshold_results:
    print(f"Umbral: {result['threshold']}")
    print(f"F1 Score: {result['f1_score']}")
    print(f"ROC-AUC: {result['roc_auc']}")
    print(f"Matriz de confusión:\n{result['confusion_matrix']}\n")

# 2. Curva Precision-Recall (PR)
precision, recall, _ = precision_recall_curve(y_test, proba)
plt.figure()
plt.plot(recall, precision, marker='.')
plt.title('Curva Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# 3. Curva ROC
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# 4. Error Medio Cuadrático (EMC) y Coeficiente de determinación (R2)
mse = mean_squared_error(y_test, y_pred_best)
r2 = best_model.score(X_test_scaled, y_test)  # Coeficiente de determinación

print(f"Error Medio Cuadrático (EMC): {mse}")
print(f"Coeficiente de determinación (R2): {r2}")

# 5. Error Absoluto Medio (EAM)
mae = mean_absolute_error(y_test, y_pred_best)
print(f"Error Absoluto Medio (EAM): {mae}")


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
# 
# Llegando al final del proyecto quiero felicitarte por esta última implementación, hemos profundizado al probar diferentes valores/umbrales para definir la predicción mediante la probabilidad. A la vez excelente puesta común de las curvas auc_roc. Felicitaciones!

# Conclusiones del Proyecto
# Modelo Inicial: El modelo inicial sin ajustes presenta una precisión aceptable en la clase mayoritaria (no salida) pero pobre en la clase minoritaria (salida). La puntuación F1 es moderada debido al desequilibrio de clases.
# 
# Modelo con Ajuste de Peso de Clase: El ajuste de peso de clase mejora ligeramente el rendimiento del modelo en términos de balancear las clases, pero la puntuación F1 sigue siendo subóptima.
# 
# Mejor Modelo Optimizado: La optimización de hiperparámetros mejora significativamente la puntuación AUC-ROC y la puntuación F1. El mejor modelo logra un equilibrio más adecuado entre precisión y recall para la clase minoritaria.
# 
# Evaluación Avanzada: La evaluación de diferentes umbrales permite ajustar el modelo para obtener una mejor relación entre precisión y recall, dependiendo de las necesidades específicas del negocio.
# 
# En resumen, el modelo final optimizado cumple con el requisito del proyecto de alcanzar un valor F1 superior a 0.59 y presenta un AUC-ROC competitivo, indicando una buena capacidad predictiva para determinar si un cliente dejará el banco.

# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)</b> <a class="tocSkip"></a>
#     
# Nuevamente quiero felicitarte Pia, tanto tus conclusiones como el procedimiento en general ha sido perfecto, demuestra una gran comprensión de lo aplicado y lo obtenido. Impresionante!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




