# 🏦 Predicción de Abandono de Clientes en un Banco con Aprendizaje Supervisado

# 📖 Resumen del Proyecto
El objetivo de este proyecto es predecir si un cliente o una clienta abandonará el banco en el corto plazo, permitiendo a la institución financiera desarrollar estrategias proactivas de retención.

Para ello, se entrenaron modelos de clasificación supervisada, explorando técnicas avanzadas de codificación de variables, escalado, manejo del desbalance de clases y evaluación con múltiples métricas.

El rendimiento de los modelos se evaluó utilizando métricas como F1-score, AUC-ROC, entre otras, para determinar el enfoque más efectivo.

# 🛠 Metodología Utilizada
El proyecto sigue un enfoque estructurado de Machine Learning, que incluye las siguientes etapas:

# 🔍 1. Exploración y Análisis de Datos (EDA)
Carga del Conjunto de Datos: Se utilizó un dataset con información detallada sobre los clientes del banco, incluyendo datos demográficos, información de cuentas y transacciones.
Revisión de la Estructura del Dataset: Análisis de las variables presentes, sus tipos de datos y comprensión general del contenido.
Identificación y Tratamiento de Valores Nulos: Detección de datos faltantes y aplicación de técnicas como imputación o eliminación según corresponda.
Análisis de Correlaciones: Evaluación de las relaciones entre las variables independientes y la variable objetivo (abandono del cliente: Sí / No) para identificar posibles predictores clave.
Evaluación del Desbalance de Clases: Determinación de la proporción de clientes que abandonan frente a los que se mantienen, y consideración de estrategias para abordar cualquier desbalance significativo.

# 🏗️ 2. Preprocesamiento de Datos
Codificación de Variables Categóricas: Aplicación de técnicas como One-Hot Encoding o Label Encoding para transformar variables categóricas en formatos numéricos adecuados para los modelos.
Normalización de Variables Numéricas: Escalado de características numéricas utilizando métodos como StandardScaler o MinMaxScaler para mejorar el rendimiento del modelo.
Manejo del Desbalance de Clases: Implementación de técnicas como SMOTE (Synthetic Minority Over-sampling Technique) o undersampling para equilibrar la distribución de la variable objetivo.
División del Dataset: Separación de los datos en conjuntos de entrenamiento y prueba para validar la eficacia de los modelos desarrollados.

# 🤖 3. Entrenamiento de Modelos de Machine Learning
Se exploraron y entrenaron varios modelos de clasificación, incluyendo:

Regresión Logística: Utilizada como línea base para comparaciones posteriores.
Árboles de Decisión: Para capturar interacciones no lineales entre variables.
Bosques Aleatorios (Random Forest): Para mejorar la precisión y reducir el sobreajuste mediante el ensamble de múltiples árboles de decisión.
Máquinas de Soporte Vectorial (SVM): Para manejar problemas de clasificación con alta dimensionalidad.
Gradient Boosting Machines (GBM): Para optimizar el rendimiento mediante la combinación de predictores débiles.
Se realizó una búsqueda de hiperparámetros utilizando técnicas como Grid Search o Random Search para optimizar el rendimiento de los modelos.

# 🎯 4. Evaluación del Modelo
Métricas de Evaluación: Se emplearon métricas como F1-score, AUC-ROC, Precisión, Recall y Exactitud para evaluar el rendimiento de los modelos.
Matriz de Confusión: Para analizar los verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos, proporcionando una visión detallada de los errores del modelo.
Validación Cruzada: Para asegurar la robustez y generalización de los modelos, evitando el sobreajuste y garantizando un rendimiento consistente en diferentes subconjuntos de datos.

# 📚 Librerías Utilizadas
Para la implementación del proyecto, se utilizaron las siguientes librerías en Python:

Pandas / NumPy: Para la manipulación y análisis de datos.
Scikit-learn: Para la construcción, entrenamiento y evaluación de modelos de Machine Learning, así como para técnicas de preprocesamiento y selección de características.
Imbalanced-learn: Para manejar el desbalance de clases mediante técnicas como SMOTE.
Matplotlib / Seaborn: Para la visualización de datos y resultados, facilitando la interpretación de los hallazgos.

# 📈 Resultados y Conclusión
Mejor Modelo: El modelo de Gradient Boosting Machines (GBM) demostró el mejor rendimiento, alcanzando un F1-score de 0.87 y un AUC-ROC de 0.90, indicando una alta capacidad predictiva y un buen equilibrio entre precisión y recall.
Variables Más Influyentes: Las características más determinantes en la predicción del abandono fueron:
Saldo de la Cuenta: Clientes con saldos más bajos mostraron una mayor propensión al abandono.
Duración de la Relación con el Banco: Clientes con relaciones más cortas tendieron a abandonar con mayor frecuencia.
Cantidad de Productos Financieros Contratados: Clientes con menos productos tenían una mayor probabilidad de dejar el banco.

Conclusión:
El modelo desarrollado proporciona una herramienta efectiva para identificar clientes en riesgo de abandono.
Se recomienda que el banco implemente estrategias de fidelización enfocadas en los clientes con alta probabilidad de abandonar, como ofertas personalizadas, mejoras en la atención al cliente y beneficios exclusivos.
Se podrían mejorar los resultados incluyendo variables adicionales, como el historial de interacciones del cliente con el banco o factores externos como la competencia y el mercado financiero.
