# 🎬 Clasificación de Reseñas de Películas con Machine Learning

# 📖 Resumen del Proyecto
Este proyecto tiene como objetivo clasificar automáticamente reseñas de películas en positivas o negativas utilizando modelos de machine learning y procesamiento de lenguaje natural (NLP).

Se entrenaron distintos modelos para analizar el sentimiento de los textos y determinar la mejor estrategia para predecir correctamente si una reseña es favorable o desfavorable.

El principal criterio de evaluación es la métrica F1-score, con un objetivo mínimo de 0.85.

# 🛠 Metodología Utilizada

Para lograr la clasificación de reseñas de películas, se siguió el siguiente proceso:

# 🔍 1. Exploración y Preparación de Datos
Carga del dataset de reseñas de películas de IMDb.
Revisión de estructura del dataset y distribución de clases.
Análisis de valores nulos y eliminación de datos innecesarios.
Análisis de la cantidad de palabras en cada reseña y su impacto en la clasificación.

# 🏗️ 2. Preprocesamiento de Texto
Para transformar los textos en datos utilizables por los modelos de machine learning, se aplicaron técnicas de Procesamiento de Lenguaje Natural (NLP), tales como:

Conversión del texto a minúsculas.
Eliminación de caracteres especiales y puntuación.
Tokenización y eliminación de stopwords (palabras irrelevantes como "el", "de", "y").
Lematización para reducir palabras a su forma base.
Representación numérica de los textos mediante:
TF-IDF (Term Frequency - Inverse Document Frequency)
Embeddings de palabras (Word2Vec, GloVe, BERT - en pruebas avanzadas)

# 🤖 3. Entrenamiento de Modelos de Machine Learning
Se probaron distintos algoritmos de clasificación para identificar el más preciso:

Regresión Logística → Modelo base para evaluar desempeño inicial.
Naïve Bayes → Modelo probabilístico efectivo para NLP.
Random Forest → Modelo basado en árboles de decisión para mejorar la clasificación.
Gradient Boosting (XGBoost, LightGBM) → Modelos avanzados para optimización de predicciones.
Redes Neuronales con Embeddings (BERT - Prueba Opcional) → Para capturar significado contextual en textos largos.

# 🎯 4. Evaluación del Modelo
Comparación de modelos utilizando métricas como Precisión, Recall y F1-score.
Cálculo de la matriz de confusión para analizar errores en la clasificación.
Optimización de hiperparámetros mediante Grid Search y Random Search.
Análisis de las palabras más influyentes en las predicciones del modelo.

# 📚 Librerías Utilizadas
Para la implementación del proyecto, se usaron diversas librerías en Python, incluyendo:

Pandas / NumPy → Manipulación y análisis de datos.
Matplotlib / Seaborn → Visualización de datos.
Scikit-learn → Modelos de machine learning y métricas de evaluación.
NLTK / SpaCy → Procesamiento de lenguaje natural.
TF-IDF / Word2Vec / BERT → Métodos de vectorización de texto.

# 📈 Resultados y Conclusión
El modelo de Regresión Logística con TF-IDF obtuvo el mejor equilibrio entre precisión y velocidad, logrando un F1-score de 0.87, superando el objetivo del proyecto.
Naïve Bayes también mostró buen desempeño, pero con menor precisión en ciertas reseñas con lenguaje más ambiguo.
Los modelos basados en embeddings (BERT) mejoraron la comprensión de contexto, pero fueron más costosos computacionalmente.
Las palabras más relevantes en la clasificación de reseñas negativas incluían términos como "aburrida", "decepcionante", "mala", mientras que en reseñas positivas destacaban "excelente", "fascinante", "maravillosa".
Se recomienda seguir optimizando los modelos con embeddings preentrenados y técnicas de transferencia de aprendizaje para mejorar la clasificación en reseñas con ironía o lenguaje subjetivo.

