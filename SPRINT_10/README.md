# 🛢️ Análisis Predictivo para la Exploración de Petróleo en OilyGiant

# 📖 Resumen del Proyecto
La empresa OilyGiant busca optimizar sus inversiones en exploración de petróleo mediante el uso de modelos de Machine Learning para predecir qué regiones tienen mayores probabilidades de contener yacimientos rentables. El objetivo principal es desarrollar un modelo predictivo que, basado en datos geológicos y de perforación, identifique las áreas más prometedoras para la extracción de petróleo, maximizando la rentabilidad y minimizando los riesgos financieros.

# 🛠 Metodología Utilizada
Para alcanzar los objetivos del proyecto, se siguió una metodología estructurada en varias etapas:

# 🔍 1. Exploración y Análisis de Datos (EDA)
Carga del Dataset: Se recopiló información geológica y de perforación de diversas regiones.
Análisis Descriptivo: Se evaluó la distribución de las variables numéricas y se identificaron posibles valores atípicos.
Correlación de Variables: Se analizaron las relaciones entre las características geológicas y el volumen de petróleo extraído.
Comparación Regional: Se compararon los datos entre diferentes regiones para evaluar diferencias en la extracción y características geológicas.

# 🏗️ 2. Preprocesamiento de Datos
Manejo de Valores Nulos: Se implementaron estrategias para tratar datos faltantes o inconsistentes.
Codificación de Variables Categóricas: Se aplicaron técnicas como One-Hot Encoding para transformar variables categóricas en numéricas.
Escalado de Variables: Se normalizaron las variables numéricas para mejorar el rendimiento de los modelos.
División de Datos: El conjunto de datos se dividió en conjuntos de entrenamiento y prueba para validar los modelos.

# 🤖 3. Desarrollo y Entrenamiento de Modelos
Se evaluaron varios modelos de regresión para predecir la cantidad de barriles de petróleo extraídos en diferentes ubicaciones:

Regresión Lineal: Como modelo base para establecer una línea de referencia.
Regresión Ridge y Lasso: Para manejar la multicolinealidad y seleccionar variables relevantes.
Árboles de Decisión y Random Forest: Para capturar relaciones no lineales y mejorar la precisión de las predicciones.
Gradient Boosting: Para optimizar el rendimiento del modelo mediante el ensamble de predictores débiles.
Se realizó una búsqueda de hiperparámetros utilizando Grid Search para optimizar el rendimiento de los modelos.

# 🎯 4. Evaluación del Modelo
Métricas de Evaluación: Se utilizaron métricas como el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²) para evaluar el rendimiento de los modelos.
Validación Cruzada: Se implementó la validación cruzada para asegurar la generalización de los modelos y prevenir el sobreajuste.
Análisis de Residuos: Se analizaron los residuos de las predicciones para identificar patrones no capturados por los modelos.

# 📚 Librerías Utilizadas
Para la implementación del proyecto, se emplearon las siguientes librerías en Python:

Pandas / NumPy: Para la manipulación y análisis de datos.
Matplotlib / Seaborn: Para la visualización de datos y resultados.
Scikit-learn: Para la construcción y evaluación de modelos de Machine Learning.
SciPy / Statsmodels: Para análisis estadísticos adicionales y pruebas de hipótesis.

# 📈 Resultados y Conclusión
Mejor Modelo: El modelo de Random Forest demostró el mejor rendimiento, con un R² elevado y un MSE bajo, indicando una alta precisión en las predicciones.
Variables Relevantes: Las características geológicas como la porosidad, permeabilidad y profundidad de las formaciones fueron las más influyentes en las predicciones.

Conclusión: El modelo desarrollado proporciona una herramienta valiosa para OilyGiant en la toma de decisiones sobre inversiones en exploración, permitiendo identificar regiones con alto potencial de extracción de petróleo y optimizar los recursos financieros.






