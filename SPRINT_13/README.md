# 🚖 Análisis de Viajes y Predicción de Demanda para Sweet Lift Taxi

# 📖 Resumen del Proyecto

El objetivo de este proyecto es analizar los patrones de viaje de Sweet Lift Taxi, una empresa de transporte por aplicación, y desarrollar un modelo de machine learning que permita predecir la demanda de viajes en función de diversas variables.

Se busca entender qué factores influyen en la cantidad de viajes y cómo optimizar la asignación de conductores para mejorar la eficiencia del servicio.

# 🛠 Metodología Utilizada
El proyecto sigue una metodología estructurada para el análisis y modelado de datos:

# 🔍 1. Exploración y Limpieza de Datos
Carga del conjunto de datos de viajes y revisión de su estructura.
Análisis de valores nulos y tratamiento de datos inconsistentes.
Conversión de formatos de fechas y variables categóricas.
Generación de nuevas variables relevantes para el análisis (e.g., día de la semana, hora del día).

# 📊 2. Análisis Exploratorio de Datos (EDA)
Distribución de la cantidad de viajes por día y por hora.
Identificación de patrones temporales en la demanda.
Análisis del impacto de factores externos, como el clima o eventos especiales.
Visualización de tendencias y correlaciones entre variables.

# 🏗️ 3. Preprocesamiento y Transformación de Datos
Codificación de variables categóricas mediante One-Hot Encoding.
Normalización de variables numéricas para mejorar el rendimiento del modelo.
Creación de un conjunto de entrenamiento y prueba.

# 🤖 4. Entrenamiento de Modelos de Machine Learning
Se probaron y compararon varios modelos de predicción, incluyendo:

Regresión Lineal → Para establecer una línea base en la predicción.
Árboles de Decisión → Para capturar relaciones no lineales en los datos.
Random Forest → Modelo basado en ensambles para mejorar precisión.
Gradient Boosting (LightGBM/XGBoost) → Modelos avanzados para optimizar predicciones.
Se realizó tuneo de hiperparámetros utilizando Grid Search y Random Search para mejorar el rendimiento del modelo.

# 🎯 5. Evaluación del Modelo
Cálculo de métricas de desempeño: RMSE (Error Cuadrático Medio) y R² (Coeficiente de Determinación).
Comparación de modelos y selección del más eficiente para la predicción de demanda.

# 📚 Librerías Utilizadas
El proyecto fue desarrollado en Python utilizando las siguientes librerías:

Pandas → Manipulación y limpieza de datos.
NumPy → Operaciones matemáticas y manejo de arreglos.
Matplotlib / Seaborn → Visualización de datos.
Scikit-learn → Modelos de machine learning, métricas y optimización.
LightGBM / XGBoost → Modelos avanzados de boosting para predicción.

# 📈 Resultados y Conclusión
El modelo de Random Forest obtuvo la mejor precisión en la predicción de la demanda de viajes, con un RMSE bajo y un alto R².
Los patrones de demanda muestran picos en horas punta (mañana y tarde), especialmente entre 7:00-9:00 AM y 5:00-7:00 PM.
El clima influye significativamente en la cantidad de viajes, con un aumento en la demanda en días de lluvia.
Se recomienda ajustar la asignación de conductores en función de estos patrones para reducir tiempos de espera y mejorar la experiencia del usuario.
