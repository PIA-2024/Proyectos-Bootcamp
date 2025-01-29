# 🏦 Predicción de Abandono de Clientes en un Banco con Aprendizaje Supervisado

# 📖 Resumen del Proyecto
El objetivo de este proyecto es predecir si un cliente o una clienta abandonará el banco en el corto plazo, permitiendo a la empresa financiera desarrollar estrategias de retención.

Para lograrlo, se entrenaron modelos de clasificación supervisada, explorando técnicas avanzadas de codificación de variables, escalado, manejo del desbalance de clases y evaluación con métricas múltiples.

El rendimiento del modelo se evaluará con F1-score, AUC-ROC y otras métricas para determinar el mejor enfoque.

# 🛠 Metodología Utilizada
El proyecto sigue un enfoque estructurado de machine learning:

# 🔍 1. Exploración y Análisis de Datos (EDA)
Carga del conjunto de datos con información sobre clientes del banco.
Revisión de la estructura del dataset, tipos de variables y distribución de datos.
Identificación de valores nulos y tratamiento de datos inconsistentes.
Análisis de correlaciones entre variables y la variable objetivo (abandono del banco: Sí / No).
Evaluación del desbalance de clases y estrategias para abordarlo.

# 🏗️ 2. Preprocesamiento de Datos
Conversión de datos categóricos mediante One-Hot Encoding y Label Encoding.
Normalización de variables numéricas con StandardScaler o MinMaxScaler.
Manejo del desbalance de clases con técnicas como undersampling, oversampling y SMOTE.
División del dataset en conjunto de entrenamiento (80%) y prueba (20%).

# 🤖 3. Entrenamiento de Modelos de Machine Learning
Se probaron distintos modelos de clasificación, incluyendo:

Regresión Logística → Modelo base para evaluar el rendimiento inicial.
Árboles de Decisión → Para capturar relaciones no lineales en los datos.
Random Forest → Modelo basado en ensambles para mejorar la predicción.
Gradient Boosting (XGBoost, LightGBM, CatBoost) → Modelos avanzados para optimización de predicciones.
Se realizó tuneo de hiperparámetros mediante Grid Search y Random Search para mejorar la precisión del modelo.

# 🎯 4. Evaluación del Modelo
Comparación de modelos utilizando métricas clave:
F1-score → Métrica principal para evaluar la clasificación.
AUC-ROC → Para medir la capacidad de diferenciación del modelo.
Precisión y Recall → Evaluación del equilibrio entre falsos positivos y negativos.
Matriz de confusión para visualizar errores de clasificación.

# 📚 Librerías Utilizadas
Para la implementación del modelo, se usaron las siguientes librerías en Python:

Pandas / NumPy → Manipulación y análisis de datos.
Matplotlib / Seaborn → Visualización de datos.
Scikit-learn → Modelos de clasificación, métricas y optimización.
Imbalanced-learn → Manejo del desbalance de clases con SMOTE.
XGBoost / LightGBM / CatBoost → Algoritmos avanzados de boosting para mejorar la clasificación.

# 📈 Resultados y Conclusión
El modelo de Gradient Boosting (XGBoost) logró el mejor rendimiento, alcanzando un F1-score alto y un AUC-ROC superior a 0.85, lo que indica una excelente capacidad de predicción.
Las variables más influyentes en la predicción del abandono fueron:
Saldo promedio del cliente.
Número de productos contratados.
Tiempo de antigüedad en el banco.
Tipo de cuenta y región del cliente.
El manejo del desbalance de clases con SMOTE mejoró la capacidad predictiva del modelo, reduciendo falsos negativos.
Se recomienda implementar estrategias de retención para los clientes con alta probabilidad de abandonar, como mejoras en el servicio y ofertas personalizadas.
