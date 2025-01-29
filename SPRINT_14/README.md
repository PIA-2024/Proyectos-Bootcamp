# 📊 Optimización del Sistema de Recomendaciones para Sweet Lift Taxi

# 📖 Resumen del Proyecto
En este proyecto, se desarrolló un sistema de recomendaciones para la empresa de transporte Sweet Lift Taxi con el objetivo de mejorar la experiencia del usuario y aumentar la fidelización de clientes.

Se analizaron datos históricos de viajes y preferencias de los clientes para implementar un modelo que personaliza sugerencias de rutas, horarios y posibles destinos basados en el comportamiento previo de cada usuario.

# 🛠 Metodología Utilizada
El proyecto siguió una metodología estructurada en varias etapas:

# 🔍 1. Exploración y Limpieza de Datos
Carga y revisión de la estructura del dataset.
Identificación y tratamiento de valores nulos e inconsistencias.
Conversión de formatos de fecha y normalización de variables categóricas.

# 📊 2. Análisis Exploratorio de Datos (EDA)
Análisis de patrones de viaje de los clientes.
Identificación de clientes recurrentes y su comportamiento.
Estudio de correlaciones entre variables como frecuencia de viaje, ubicación y tipo de servicio solicitado.
Visualización de tendencias de uso en distintas horas y días de la semana.

# 🏗️ 3. Ingeniería de Características y Preprocesamiento
Creación de variables relevantes, como tipo de usuario (frecuente, ocasional), horario preferido y distancia promedio de viaje.
Codificación de variables categóricas mediante One-Hot Encoding y Label Encoding.
Normalización de variables numéricas para mejorar el rendimiento del modelo.

# 🤖 4. Implementación del Sistema de Recomendaciones
Se exploraron distintos enfoques para generar recomendaciones:

Modelo basado en reglas → Sugerencias personalizadas en función del historial de viajes.
Filtrado Colaborativo → Basado en similitud entre usuarios con patrones de viaje similares.
Filtrado Basado en Contenido → Uso de características del usuario para predecir sus preferencias futuras.
Modelos híbridos → Combinación de enfoques para mejorar la precisión de las recomendaciones.

# 🎯 5. Evaluación del Sistema de Recomendaciones
Se utilizaron métricas como Precisión@K, Recall@K y NDCG para evaluar la calidad de las recomendaciones.
Comparación entre distintos métodos para identificar el más eficiente.
Ajuste de hiperparámetros y optimización del modelo final.

# 📚 Librerías Utilizadas
Para la implementación del sistema de recomendaciones, se utilizaron las siguientes librerías en Python:

Pandas → Análisis y manipulación de datos.
NumPy → Operaciones matemáticas.
Matplotlib / Seaborn → Visualización de datos.
Scikit-learn → Modelos de machine learning y métricas de evaluación.
Surprise → Algoritmos de recomendación como SVD y KNN.

# 📈 Resultados y Conclusión
Se logró implementar un sistema de recomendaciones eficiente, mejorando la personalización del servicio de Sweet Lift Taxi.
El modelo híbrido fue el más efectivo, combinando técnicas de filtrado colaborativo y basado en contenido.
Los clientes frecuentes mostraron mayor tasa de conversión en las recomendaciones en comparación con los usuarios esporádicos.
Se recomienda seguir mejorando el sistema con aprendizaje profundo y modelos secuenciales para predecir patrones más complejos de comportamiento.


