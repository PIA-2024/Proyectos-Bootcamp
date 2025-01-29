# 🛢️ Análisis Predictivo para la Exploración de Petróleo en OilyGiant

# 📖 Resumen del Proyecto
La empresa OilyGiant busca optimizar sus inversiones en exploración de petróleo utilizando modelos de machine learning para predecir qué regiones tienen mayores probabilidades de contener yacimientos rentables.

El objetivo principal del proyecto es desarrollar un modelo predictivo que, basado en datos geológicos y de perforación, pueda identificar las regiones más prometedoras para la extracción de petróleo, maximizando la rentabilidad y minimizando los riesgos financieros.

Se evaluaron distintos modelos de regresión para predecir la cantidad de barriles de petróleo extraídos en diferentes ubicaciones.

# 🛠 Metodología Utilizada
El proyecto se dividió en varias etapas clave:

# 🔍 1. Exploración y Análisis de Datos (EDA)
Carga y revisión del dataset con información geológica de distintas regiones.
Análisis de la distribución de las variables numéricas y detección de valores atípicos.
Identificación de correlaciones entre características geológicas y el volumen de petróleo extraído.
Comparación de datos entre distintas regiones para evaluar diferencias en la extracción.

# 🏗️ 2. Preprocesamiento de Datos
Manejo de valores nulos y datos inconsistentes.
Escalado de variables numéricas para mejorar el rendimiento del modelo.
División del dataset en conjunto de entrenamiento (80%) y prueba (20%).

# 🤖 3. Entrenamiento de Modelos Predictivos
Se probaron varios modelos de regresión supervisada para predecir el volumen de petróleo extraído en una ubicación dada:

Regresión Lineal → Modelo base para evaluar desempeño inicial.
Árboles de Decisión → Para capturar relaciones no lineales en los datos.
Random Forest → Modelo basado en ensambles para mejorar predicciones.
Gradient Boosting (XGBoost, LightGBM) → Modelos avanzados para optimización de predicciones.

# 🎯 4. Evaluación del Modelo
Comparación de modelos utilizando métricas como:
RMSE (Error Cuadrático Medio) → Para medir la precisión de la predicción.
R² (Coeficiente de Determinación) → Para evaluar el ajuste del modelo.
Análisis de la importancia de variables en la predicción del volumen de petróleo.
Selección del modelo más eficiente en función del trade-off entre precisión y costo computacional.

# 📚 Librerías Utilizadas
Para la implementación del modelo de predicción, se utilizaron las siguientes librerías en Python:

Pandas / NumPy → Manipulación y análisis de datos.
Matplotlib / Seaborn → Visualización de datos.
Scikit-learn → Modelos de regresión, métricas y optimización.
LightGBM / XGBoost → Algoritmos avanzados de boosting para mejorar la predicción.

# 📈 Resultados y Conclusión
El modelo de Random Forest obtuvo el mejor rendimiento, logrando un RMSE bajo y un R² alto, lo que indica una buena capacidad de predicción.
Las características más relevantes para predecir la extracción de petróleo fueron:
Presión del suelo en la región.
Densidad del petróleo extraído.
Profundidad de perforación.
Se recomienda utilizar el modelo para evaluar nuevas regiones antes de invertir en perforaciones costosas, ayudando a minimizar pérdidas y optimizar la exploración.
Futuras mejoras podrían incluir la integración de modelos de Deep Learning para mejorar aún más la predicción.









