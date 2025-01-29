# 🎬 Clasificación de Reseñas de Películas con Machine Learning

# 📖 Resumen del Proyecto
Este proyecto tiene como objetivo clasificar automáticamente reseñas de películas en positivas o negativas utilizando modelos de Machine Learning y técnicas de Procesamiento de Lenguaje Natural (NLP). Se entrenaron distintos modelos para analizar el sentimiento de los textos y determinar la mejor estrategia para predecir correctamente si una reseña es favorable o desfavorable. El principal criterio de evaluación es la métrica F1-score, con un objetivo mínimo de 0.85.

# 🛠 Metodología Utilizada
Para lograr la clasificación de reseñas de películas, se siguió el siguiente proceso:

# 🔍 1. Exploración y Preparación de Datos
Carga del Dataset: Se utilizó el conjunto de datos de reseñas de películas de IMDb, que contiene textos de reseñas junto con su clasificación como positivas o negativas.
Revisión de la Estructura del Dataset: Análisis de la cantidad de reseñas, distribución de clases y verificación de la presencia de valores nulos o datos faltantes.
Análisis de la Longitud de las Reseñas: Estudio de la cantidad de palabras en cada reseña y su impacto en la clasificación.

# 🏗️ 2. Preprocesamiento de Texto
Para transformar los textos en datos utilizables por los modelos de machine learning, se aplicaron técnicas de NLP, tales como:

Limpieza del Texto: Conversión del texto a minúsculas, eliminación de caracteres especiales, puntuación y números.
Tokenización: División del texto en palabras o tokens individuales.
Eliminación de Stop Words: Remoción de palabras comunes que no aportan significado significativo al análisis (e.g., "el", "la", "de").
Lematización/Stemming: Reducción de las palabras a su forma base o raíz.
Vectorización: Conversión de los textos preprocesados en representaciones numéricas utilizando técnicas como TF-IDF (Term Frequency-Inverse Document Frequency).

# 🤖 3. Entrenamiento de Modelos de Machine Learning
Se probaron distintos modelos de clasificación para determinar cuál ofrecía el mejor rendimiento:

Regresión Logística: Modelo base para evaluar el rendimiento inicial.
Máquinas de Soporte Vectorial (SVM): Para capturar relaciones complejas en los datos.
Árboles de Decisión y Random Forest: Modelos basados en ensambles para mejorar la predicción.
Naive Bayes: Modelo probabilístico comúnmente utilizado en clasificación de texto.
Se realizó ajuste de hiperparámetros mediante Grid Search para optimizar el rendimiento de los modelos.

# 🎯 4. Evaluación del Modelo
Métricas de Evaluación: Se utilizaron métricas como F1-score, Precisión, Recall y Exactitud para evaluar el rendimiento de los modelos.
Matriz de Confusión: Para analizar los verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
Validación Cruzada: Para asegurar la generalización del modelo y evitar sobreajuste.

# 📚 Librerías Utilizadas
Para la implementación del proyecto, se utilizaron las siguientes librerías en Python:

Pandas / NumPy: Manipulación y análisis de datos.
NLTK / SpaCy: Herramientas para procesamiento de lenguaje natural.
Scikit-learn: Modelos de clasificación, métricas y herramientas de preprocesamiento.
Matplotlib / Seaborn: Visualización de datos y resultados.

# 📈 Resultados y Conclusión
Mejor Modelo: El modelo de Máquinas de Soporte Vectorial (SVM) con kernel lineal alcanzó un F1-score de 0.87, superando el objetivo establecido de 0.85.
Características Relevantes: Las palabras con mayor peso en la clasificación fueron términos claramente asociados con sentimientos positivos o negativos, lo que indica que el modelo capturó adecuadamente el sentimiento de las reseñas.

Conclusión: El modelo desarrollado es efectivo para clasificar automáticamente reseñas de películas en positivas o negativas, y puede ser implementado en sistemas de recomendación o análisis de opinión para mejorar la experiencia del usuario.
