Proyecto: Predicción de Deserción de Clientes para Beta Bank
Este proyecto tiene como objetivo predecir si un cliente dejará Beta Bank en un futuro cercano. Utilizando datos históricos sobre el comportamiento de los clientes, se entrenó un modelo de machine learning para maximizar la métrica F1 y evaluar el rendimiento mediante AUC-ROC.

Descripción del Proyecto
Objetivos principales:

Preparación y Análisis de Datos: Descargar y preprocesar los datos para asegurar que estén listos para el modelado.
Entrenamiento Inicial del Modelo: Entrenar un modelo inicial sin ajustar el desequilibrio de clases para establecer una línea base.
Manejo del Desequilibrio de Clases: Implementar al menos dos enfoques para corregir el desequilibrio de clases y mejorar la calidad del modelo.
Optimización de Hiperparámetros: Utilizar GridSearchCV para encontrar el mejor modelo y conjunto de parámetros.
Evaluación Final: Realizar pruebas finales en el conjunto de prueba y comparar las métricas F1 y AUC-ROC.
Estructura del Proyecto
Preparación de Datos:

Se eliminaron columnas innecesarias como RowNumber, CustomerId, y Surname.
Se aplicó la codificación One-Hot para las variables categóricas como Geography y Gender.
Se realizó el escalado de características numéricas utilizando StandardScaler.
Entrenamiento Inicial del Modelo:

Se entrenó un modelo de Árbol de Decisión como una línea base sin ajustes para el desequilibrio de clases.
Se evaluó el rendimiento inicial utilizando la métrica F1 y AUC-ROC.
Manejo del Desequilibrio de Clases:

Se implementó el ajuste de pesos en el modelo de Árbol de Decisión para corregir el desequilibrio de clases.
Se compararon los resultados del modelo ajustado con la línea base.
Optimización de Hiperparámetros:

Se utilizó GridSearchCV para optimizar los hiperparámetros del modelo y seleccionar el mejor estimador.
Se evaluó el rendimiento del mejor modelo en el conjunto de prueba.
Evaluación Avanzada:

Se evaluó el rendimiento del modelo en diferentes umbrales de decisión.
Se generaron las curvas Precision-Recall y ROC para una evaluación detallada.

Preparación de Datos:

Carga los datos y realiza el preprocesamiento, incluyendo la eliminación de columnas innecesarias, la codificación de variables categóricas y el escalado de características.
Entrenamiento y Evaluación del Modelo:

Entrena el modelo utilizando los datos preprocesados.
Evalúa el modelo utilizando las métricas F1 y AUC-ROC.
Implementa estrategias para manejar el desequilibrio de clases y optimiza los hiperparámetros.
Evaluación Avanzada:

Ajusta los umbrales de decisión para maximizar el rendimiento según las necesidades del negocio.
Genera y analiza las curvas Precision-Recall y ROC.
Tecnologías Utilizadas
Python: Lenguaje principal utilizado en el proyecto.
Pandas: Para manipulación y análisis de datos.
Scikit-learn: Para la implementación de algoritmos de machine learning y evaluación de modelos.
Matplotlib: Para visualización de datos.
¿Por qué elegimos estas tecnologías?
Pandas proporciona una manera eficiente de manejar y preprocesar grandes conjuntos de datos, mientras que Scikit-learn ofrece un amplio conjunto de herramientas para entrenar y evaluar modelos de machine learning. Matplotlib es ideal para visualizar los resultados de los análisis y evaluar el rendimiento del modelo.

Conclusiones
Modelo Inicial: El modelo inicial presentó un rendimiento moderado con un F1 score que se vio afectado por el desequilibrio de clases.
Ajuste de Peso de Clase: La corrección del desequilibrio de clases mediante el ajuste de pesos mejoró la precisión en la clase minoritaria, aunque no fue suficiente.
Optimización de Hiperparámetros: La optimización con GridSearchCV mejoró significativamente el rendimiento del modelo, logrando un equilibrio adecuado entre precisión y recall.
Evaluación Avanzada: Ajustar los umbrales de decisión permite maximizar la relación entre precisión y recall, adaptando el modelo a las necesidades específicas del negocio.
