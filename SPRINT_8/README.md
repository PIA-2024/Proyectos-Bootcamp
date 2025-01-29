# 📱 Clasificación de Clientes para Planes Tarifarios de Megaline

# 📖 Resumen del Proyecto
La empresa de telecomunicaciones Megaline busca optimizar la migración de sus clientes desde planes heredados hacia sus nuevos planes tarifarios: Smart y Ultra.

El objetivo de este proyecto es desarrollar un modelo de clasificación supervisada que, basado en el comportamiento y características de los clientes, recomiende el plan más adecuado para cada usuario.

El modelo se evaluará utilizando la métrica de exactitud (accuracy), con un umbral mínimo de 0.75 para ser considerado satisfactorio.

# 🛠 Metodología Utilizada
El desarrollo del proyecto se estructuró en las siguientes etapas:

# 🔍 1. Exploración y Análisis de Datos (EDA)
Carga del Dataset: Se recopiló información detallada sobre el uso de servicios por parte de los clientes, incluyendo datos como la cantidad de minutos de llamadas, mensajes de texto enviados y volumen de datos utilizados.
Análisis Descriptivo: Se evaluó la distribución de variables numéricas y categóricas para comprender el comportamiento general de los clientes.
Detección de Valores Nulos: Se identificaron y analizaron valores faltantes, determinando estrategias de imputación o eliminación según corresponda.
Análisis de Correlación: Se investigaron las relaciones entre las variables independientes y la variable objetivo (plan tarifario: Smart o Ultra) para identificar posibles predictores clave.

# 🏗️ 2. Preprocesamiento de Datos
Codificación de Variables Categóricas: Se aplicaron técnicas como One-Hot Encoding o Label Encoding para transformar variables categóricas en formatos numéricos adecuados para los modelos de Machine Learning.
Normalización de Variables Numéricas: Se realizó el escalado de características numéricas utilizando métodos como StandardScaler o MinMaxScaler para mejorar el rendimiento y la convergencia de los modelos.
División del Dataset: Se separaron los datos en conjuntos de entrenamiento y prueba para validar la eficacia de los modelos desarrollados.

# 🤖 3. Entrenamiento de Modelos de Machine Learning
Se exploraron y entrenaron varios modelos de clasificación, incluyendo:

Regresión Logística: Utilizada como línea base para comparaciones posteriores.
Árboles de Decisión: Para capturar interacciones no lineales entre variables y facilitar la interpretación de los resultados.
Bosques Aleatorios (Random Forest): Para mejorar la precisión y reducir el sobreajuste mediante el ensamble de múltiples árboles de decisión.
Máquinas de Soporte Vectorial (SVM): Para manejar problemas de clasificación con alta dimensionalidad y encontrar el hiperplano óptimo que separa las clases.
Gradient Boosting (XGBoost / LightGBM): Modelos avanzados de boosting para optimizar las predicciones.
Se realizó una búsqueda de hiperparámetros utilizando técnicas como Grid Search o Random Search para optimizar el rendimiento de los modelos.

# 🎯 4. Evaluación del Modelo
Métricas de Evaluación: Se emplearon métricas como Exactitud (Accuracy), Precisión, Recall y F1-Score para evaluar el rendimiento de los modelos.
Matriz de Confusión: Para analizar los verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos, proporcionando una visión detallada de los errores del modelo.
Validación Cruzada: Para asegurar la robustez y generalización de los modelos, evitando el sobreajuste y garantizando un rendimiento consistente en diferentes subconjuntos de datos.

# 📚 Librerías Utilizadas
Para la implementación del proyecto, se utilizaron las siguientes librerías en Python:

Pandas / NumPy: Para la manipulación y análisis de datos.
Scikit-learn: Para la construcción, entrenamiento y evaluación de modelos de Machine Learning, así como para técnicas de preprocesamiento y selección de características.
Matplotlib / Seaborn: Para la visualización de datos y resultados, facilitando la interpretación de los hallazgos.
XGBoost / LightGBM: Modelos avanzados de boosting para optimizar la predicción de planes tarifarios.

# 📈 Resultados y Conclusión
Mejor Modelo: El modelo de Bosques Aleatorios (Random Forest) demostró el mejor rendimiento, alcanzando una exactitud (accuracy) de 0.78, superando el umbral establecido de 0.75.
Variables Más Influyentes: Las características más determinantes en la recomendación del plan fueron:
Uso de Datos Móviles: Clientes con un alto consumo de datos tendieron a ser recomendados para el plan Ultra.
Cantidad de Minutos de Llamadas: Clientes con un uso intensivo de llamadas fueron más propensos a ser recomendados para el plan Smart.
Número de Mensajes de Texto Enviados: Aunque con menor peso, también influyó en la recomendación del plan adecuado.

Conclusión:
El modelo desarrollado proporciona una herramienta efectiva para recomendar planes tarifarios a los clientes de Megaline, basándose en su comportamiento de uso.
Se recomienda que la empresa implemente este modelo para guiar a los clientes en la selección del plan que mejor se adapte a sus necesidades, lo que podría mejorar la satisfacción del cliente y optimizar el uso de los recursos de la red.
Para futuras mejoras, se podrían probar modelos más avanzados como redes neuronales o implementar sistemas de recomendación híbridos para personalizar aún más las recomendaciones.
