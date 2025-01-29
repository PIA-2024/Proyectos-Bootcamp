# 🚖 Predicción de Cancelación de Clientes en Sweet Lift Taxi

# 📖 Resumen del Proyecto
El objetivo de este proyecto es predecir la cancelación de clientes en la empresa de transporte Sweet Lift Taxi, permitiendo a la compañía identificar patrones de abandono y desarrollar estrategias para mejorar la retención de clientes.

Para lograr esto, se entrenaron modelos de clasificación supervisada, utilizando datos demográficos y de comportamiento de los clientes. Se evaluó el desempeño de los modelos utilizando métricas como AUC-ROC y F1-score para determinar la mejor estrategia de predicción.

# 🛠 Metodología Utilizada
Para desarrollar este modelo de clasificación, se siguieron los siguientes pasos:

# 🔍 1. Exploración y Análisis de Datos (EDA)
Carga del dataset con información sobre clientes de Sweet Lift Taxi.
Revisión de la estructura del dataset, tipos de variables y distribución de datos.
Identificación de valores nulos y tratamiento de datos inconsistentes.
Análisis de correlaciones entre variables y la variable objetivo (cancelación del cliente: Sí / No).

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
Gradient Boosting (XGBoost, LightGBM) → Modelos avanzados para optimización de predicciones.
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
XGBoost / LightGBM → Algoritmos avanzados de boosting para mejorar la clasificación.

# 📈 Resultados y Conclusión
El modelo de Gradient Boosting (XGBoost) logró el mejor rendimiento, alcanzando un F1-score alto y un AUC-ROC superior a 0.85, lo que indica una excelente capacidad de predicción.
Las variables más influyentes en la predicción de cancelación fueron:
Frecuencia de uso del servicio.
Método de pago utilizado.
Tiempo desde el último viaje.
Número de viajes realizados en los últimos meses.
El manejo del desbalance de clases con SMOTE mejoró la capacidad predictiva del modelo, reduciendo falsos negativos.
Se recomienda implementar estrategias de retención para los clientes con alta probabilidad de cancelar, como programas de fidelización o descuentos personalizados.
