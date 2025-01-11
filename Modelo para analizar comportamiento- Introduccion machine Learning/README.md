Modelo de Clasificación para la Asignación de Planes Móviles en Megaline
Descripción del Proyecto
La compañía móvil Megaline está interesada en migrar a sus clientes desde planes heredados a nuevos planes más adecuados. Para facilitar esta transición, desarrollamos un modelo de clasificación que analiza el comportamiento de los clientes y recomienda uno de los nuevos planes: Smart o Ultra.

Este proyecto tiene como objetivo crear un modelo de clasificación con la mayor exactitud posible para recomendar el plan correcto basado en el comportamiento de los clientes.

Propósito del Proyecto
El objetivo principal es desarrollar un modelo de clasificación con una exactitud mínima de 0.75, que pueda identificar el plan adecuado (Smart o Ultra) para los clientes de Megaline. Este modelo permitirá a la empresa optimizar sus ofertas de planes y mejorar la satisfacción del cliente.


Funcionamiento del Proyecto
Datos Utilizados
El dataset utilizado en este proyecto contiene información sobre el comportamiento mensual de los clientes. Las variables clave incluyen:

calls: número de llamadas realizadas.
minutes: duración total de las llamadas en minutos.
messages: número de mensajes de texto enviados.
mb_used: tráfico de Internet utilizado en MB.
is_ultra: plan utilizado en el mes actual (Ultra = 1, Smart = 0).
Estructura del Código
Carga y Exploración de Datos: Se realiza una exploración inicial del dataset para entender la estructura y la distribución de las variables.

Segmentación de Datos: Se divide el dataset en conjuntos de entrenamiento, validación y prueba para entrenar y evaluar los modelos.

Investigación de Modelos: Se prueban diferentes modelos de clasificación (Regresión Logística, Árbol de Decisión y Bosque Aleatorio) para identificar el que ofrece mejor rendimiento.

Ajuste de Hiperparámetros: Se ajustan los hiperparámetros del modelo más prometedor (Bosque Aleatorio) usando GridSearchCV para optimizar su rendimiento.

Evaluación Final y Prueba de Cordura: El mejor modelo se evalúa en el conjunto de prueba y se realiza una prueba de cordura para validar su comportamiento en casos extremos.

Resultados y Conclusiones
Mejor Modelo: El modelo de Bosque Aleatorio con ajuste de hiperparámetros alcanzó la mejor exactitud en validación.

Exactitud en Conjunto de Prueba: El modelo final alcanzó una exactitud de [inserte la exactitud aquí] en el conjunto de prueba, superando el umbral de 0.75.

Prueba de Cordura: El modelo muestra un comportamiento lógico en situaciones extremas, clasificando adecuadamente según el uso de datos, llamadas y mensajes.

Tecnologías Utilizadas
Python: Lenguaje de programación principal para el desarrollo del modelo.
Pandas y NumPy: Manipulación y análisis de datos.
Scikit-learn: Implementación de modelos de clasificación y ajuste de hiperparámetros.
Jupyter Notebook: Plataforma interactiva para el desarrollo y documentación del proyecto.

