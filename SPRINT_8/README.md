# 📡 Clasificación de Clientes para Planes de Megaline

# 📖 Resumen del Proyecto
Megaline, una compañía de telefonía móvil, quiere optimizar la migración de sus clientes hacia nuevos planes tarifarios. Actualmente, muchos usuarios siguen usando planes heredados, lo que genera un desafío en la adopción de los planes Smart y Ultra.

Este proyecto tiene como objetivo desarrollar un modelo de clasificación que, basado en el comportamiento de los clientes, recomiende el plan más adecuado.

El modelo se evaluará utilizando la métrica de exactitud (accuracy) y debe alcanzar un mínimo de 0.75 para ser considerado satisfactorio.

# 🛠 Metodología Utilizada
El proyecto sigue una serie de pasos clave en el desarrollo de modelos de clasificación:

# 🔍 1. Exploración y Análisis de Datos (EDA)
Carga del dataset con información del uso de servicios por los clientes.
Análisis de la distribución de variables numéricas y categóricas.
Identificación de valores nulos y estrategia de imputación.
Análisis de correlación entre variables y la variable objetivo (Plan: Smart o Ultra).

# 🏗️ 2. Preprocesamiento de Datos
Conversión de datos categóricos mediante One-Hot Encoding o Label Encoding.
Normalización de variables numéricas para mejorar la eficiencia del modelo.
Separación del dataset en conjunto de entrenamiento (80%) y prueba (20%).

# 🤖 3. Entrenamiento de Modelos de Machine Learning
Se probaron varios algoritmos de clasificación para determinar cuál ofrece el mejor rendimiento:

Regresión Logística → Modelo base para evaluar desempeño inicial.
Árboles de Decisión → Para capturar relaciones no lineales en los datos.
Random Forest → Modelo basado en ensambles para mejorar la clasificación.
Gradient Boosting (XGBoost, LightGBM) → Modelos avanzados que optimizan predicciones.

# 🎯 4. Evaluación del Modelo
Comparación de modelos utilizando métricas de clasificación:
Exactitud (accuracy) → Métrica principal, debe ser ≥ 0.75.
Matriz de confusión para analizar falsos positivos y negativos.
Precisión, Recall y F1-score para evaluar balance en la clasificación.
Ajuste de hiperparámetros mediante Grid Search o Random Search para mejorar la exactitud.

# 📚 Librerías Utilizadas
Para el desarrollo del modelo, se emplearon las siguientes librerías en Python:

Pandas / NumPy → Manipulación y análisis de datos.
Matplotlib / Seaborn → Visualización de datos.
Scikit-learn → Modelos de clasificación, métricas y optimización.
XGBoost / LightGBM → Algoritmos avanzados de boosting para mejorar la predicción.

# 📈 Resultados y Conclusión
El modelo de Random Forest alcanzó una exactitud superior al 75%, cumpliendo con los requisitos del proyecto.
Las características más importantes en la predicción fueron:
Duración de llamadas y cantidad de SMS enviados.
Uso de datos móviles en GB.
Historial de facturación y pagos.
El modelo puede ser utilizado para recomendar automáticamente planes a nuevos clientes, optimizando la retención y migración de usuarios.
Se recomienda seguir refinando el modelo con técnicas de aprendizaje profundo o sistemas de recomendación basados en aprendizaje automático.
