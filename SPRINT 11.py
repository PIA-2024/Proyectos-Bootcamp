#!/usr/bin/env python
# coding: utf-8

# ¡Hola!
# 
# Mi nombre es Tonatiuh Cruz. Me complace revisar tu proyecto hoy.
# 
# Al identificar cualquier error inicialmente, simplemente los destacaré. Te animo a localizar y abordar los problemas de forma independiente como parte de tu preparación para un rol como data-scientist. En un entorno profesional, tu líder de equipo seguiría un enfoque similar. Si encuentras la tarea desafiante, proporcionaré una pista más específica en la próxima iteración.
# 
# Encontrarás mis comentarios a continuación - **por favor no los muevas, modifiques o elimines**.
# 
# Puedes encontrar mis comentarios en cajas verdes, amarillas o rojas como esta:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Éxito. Todo está hecho correctamente.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Observaciones. Algunas recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Necesita corrección. El bloque requiere algunas correcciones. El trabajo no puede ser aceptado con comentarios en rojo.
# </div>
# 
# Puedes responderme utilizando esto:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante.</b> <a class="tocSkip"></a>
# </div>

# # Descripción

# La compañía de seguros Sure Tomorrow quiere resolver varias tareas con la ayuda de machine learning y te pide que evalúes esa posibilidad.
# - Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
# - Tarea 2: predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro. ¿Puede un modelo de predictivo funcionar mejor que un modelo dummy?
# - Tarea 3: predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir utilizando un modelo de regresión lineal.
# - Tarea 4: proteger los datos personales de los clientes sin afectar al modelo del ejercicio anterior. Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento u ofuscación de datos. Pero los datos deben protegerse de tal manera que no se vea afectada la calidad de los modelos de machine learning. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.
# 

# # Preprocesamiento y exploración de datos
# 
# ## Inicialización

# In[42]:


pip install scikit-learn --upgrade


# In[43]:


import numpy as np
import pandas as pd
import math
import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.model_selection import train_test_split

from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ## Carga de datos

# Carga los datos y haz una revisión básica para comprobar que no hay problemas obvios.

# In[44]:


df = pd.read_csv('/datasets/insurance_us.csv')


# Renombramos las columnas para que el código se vea más coherente con su estilo.

# In[45]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# In[46]:


df.sample(10)


# In[47]:


df.info()


# In[48]:


# puede que queramos cambiar el tipo de edad (de float a int) aunque esto no es crucial

# escribe tu conversión aquí si lo deseas:

# Convertir la columna 'age' de float a int
df['age'] = df['age'].astype(int)


# In[49]:


# comprueba que la conversión se haya realizado con éxito
print(df.dtypes)


# In[50]:


# ahora echa un vistazo a las estadísticas descriptivas de los datos.# ¿Se ve todo bien?


# In[51]:


print(df.describe())


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy bien! Importaste de forma correcta las librerías y módulos, cambiaste el formato de los nombres y verificaste que no hubiera outliers inválidos en los datos.
# </div>

# ## Análisis exploratorio de datos

# Vamos a comprobar rápidamente si existen determinados grupos de clientes observando el gráfico de pares.

# In[52]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# De acuerdo, es un poco complicado detectar grupos obvios (clústeres) ya que es difícil combinar diversas variables simultáneamente (para analizar distribuciones multivariadas). Ahí es donde LA y ML pueden ser bastante útiles.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelente trabajo con el desarrollo de la gráfica de pares. 
# </div>

# # Tarea 1. Clientes similares

# En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos.
# Es posible que quieras revisar las siguientes lecciones (capítulo -> lección)- Distancia entre vectores -> Distancia euclidiana
# - Distancia entre vectores -> Distancia Manhattan
# 
# Para resolver la tarea, podemos probar diferentes métricas de distancia.

# Escribe una función que devuelva los k vecinos más cercanos para un $n^{th}$ objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas.
# Puedes utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) o tu propia implementación.
# Pruébalo para cuatro combinaciones de dos casos- Escalado
#   - los datos no están escalados
#   - los datos se escalan con el escalador [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
# - Métricas de distancia
#   - Euclidiana
#   - Manhattan
# 
# Responde a estas preguntas:- ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?- ¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

# In[53]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[54]:


def get_knn(df, n, k, metric):
    
    """
    Devuelve los k vecinos más cercanos

    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar    :param n: número de objetos para los que se buscan los vecinos más cercanos    :param k: número de vecinos más cercanos a devolver
    :param métrica: nombre de la métrica de distancia    """

 # Crear el modelo kNN   
    nbrs =NearestNeighbors(n_neighbors=k, metric=metric) # <tu código aquí> 
 # Entrenar el modelo
    nbrs.fit(df[feature_names])
 # Obtener las distancias y los índices de los vecinos más cercanos
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
 # Preparar el resultado en un DataFrame    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# Escalar datos.

# In[55]:


feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[56]:


df_scaled.sample(5)


# Ahora, vamos a obtener registros similares para uno determinado, para cada combinación

# In[57]:


# Probar con datos escalados y métrica euclidiana
result_scaled_euclidean = get_knn(df_scaled, n=0, k=5, metric="euclidean")
print(result_scaled_euclidean)


# In[58]:


# Probar con datos escalados y métrica Manhattan
result_scaled_manhattan = get_knn(df_scaled, n=0, k=5, metric="manhattan")
print(result_scaled_manhattan)


# Respuestas a las preguntas

# **¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?** 
# 
# Respuesta: Sí, afecta. Cuando los datos no están escalados, las características con rangos más amplios (como el income) dominan el cálculo de las distancias, lo que puede llevar a resultados sesgados. El escalado normaliza las características para que todas tengan el mismo peso en el cálculo de la distancia.
# 

# **¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?** 
# 
# Respuesta: La métrica Manhattan tiende a ser menos influenciada por valores extremos y da más importancia a las diferencias absolutas entre las características, lo que puede llevar a vecinos ligeramente diferentes en comparación con la métrica Euclidiana.

# Código para Validación Adicional:
# Vamos a implementar este paso:

# In[59]:


# Probar con diferentes clientes
print("Vecinos más cercanos para el cliente 10 con métrica Euclidiana:")
result_scaled_euclidean_10 = get_knn(df_scaled, n=10, k=5, metric="euclidean")
print(result_scaled_euclidean_10)

print("\nVecinos más cercanos para el cliente 20 con métrica Euclidiana:")
result_scaled_euclidean_20 = get_knn(df_scaled, n=20, k=5, metric="euclidean")
print(result_scaled_euclidean_20)

# Probar con diferentes valores de k
print("\nVecinos más cercanos para el cliente 0 con métrica Euclidiana, k=3:")
result_scaled_euclidean_k3 = get_knn(df_scaled, n=0, k=3, metric="euclidean")
print(result_scaled_euclidean_k3)

print("\nVecinos más cercanos para el cliente 0 con métrica Euclidiana, k=10:")
result_scaled_euclidean_k10 = get_knn(df_scaled, n=0, k=10, metric="euclidean")
print(result_scaled_euclidean_k10)


# Conclusión de la Validación:
# Consistencia: La función get_knn es consistente y devuelve vecinos similares al cliente objetivo, lo que confirma que está funcionando correctamente.
# Variación con k: Al variar k, la función sigue siendo robusta, aunque se observa que, al aumentar k, la variabilidad en las características de los vecinos también aumenta, lo cual es normal.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Bien implementado el modelo. Como acabas de notar, el escalado es muy importante debido a que permite que variables numéricas con rangos muy diferentes sean comparables. Sin escalar los datos las características con valores más grandes pueden dominar las métricas de distancia y no obtener resultados precisos.
# </div>

# # Tarea 2. ¿Es probable que el cliente reciba una prestación del seguro?

# En términos de machine learning podemos considerarlo como una tarea de clasificación binaria.

# Con el valor de `insurance_benefits` superior a cero como objetivo, evalúa si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy.
# Instrucciones:
# - Construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Sería interesante observar cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. Puedes utilizar una implementación ya existente del algoritmo de clasificación kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) o tu propia implementación.- Construye un modelo dummy que, en este caso, es simplemente un modelo aleatorio. Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1.
# La probabilidad de pagar cualquier prestación del seguro puede definirse como
# $$
# P\{\text{prestación de seguro recibida}\}=\frac{\text{número de clientes que han recibido alguna prestación de seguro}}{\text{número total de clientes}}.
# $$
# 
# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.

# Paso 1: Definir la Variable Objetivo

# In[60]:


# сalcula el objetivo , definir la variable objetivo
df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int) #<tu código aquí>


# In[61]:


# comprueba el desequilibrio de clases con value_counts()

print(df['insurance_benefits_received'].value_counts(normalize=True))


# Paso 2: Dividir los Datos

# In[62]:


# Definir las características y la variable objetivo
X = df[['gender', 'age', 'income', 'family_members']]  # Características
y = df['insurance_benefits_received']  # Variable objetivo

# Dividir los datos en entrenamiento y prueba (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Paso 3: Evaluar el Clasificador kNN
# Aquí es donde entrenas el modelo kNN y evalúas su rendimiento. Utilizarás el código proporcionado para construir un clasificador kNN y medir la métrica F1 para diferentes valores de k

# In[63]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# si tienes algún problema con la siguiente línea, reinicia el kernel y ejecuta el cuaderno de nuevo    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Matriz de confusión')
    print(cm)
    


# In[64]:


# Entrenar y evaluar el clasificador kNN
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f'k={k}')
    eval_classifier(y_test, y_pred)


# Paso 4: Construir el Modelo Dummy
# Finalmente, construimos el modelo dummy que predice de manera aleatoria si un cliente recibirá beneficios. Este modelo sirve como referencia para comparar el rendimiento del kNN.

# In[65]:


# generar la salida de un modelo aleatorio

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[66]:


# Calcular la probabilidad observada de recibir una prestación
P_observed = df['insurance_benefits_received'].sum() / len(df)


# In[67]:


# Evaluar el modelo dummy con diferentes probabilidades
for P in [0, P_observed, 0.5, 1]:
    print(f'Probabilidad: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, size=len(y_test))
    eval_classifier(y_test, y_pred_rnd)
    print()


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelente! Tu balanceo de clases usando downsampling, separación de datos y resultados del modelo fueron muy buenos! Cuál sería tu intepretación del hecho que el escalado de datos mejore considerablemente el F1?
#     
# </div>

# # Tarea 3. Regresión (con regresión lineal)

# Con `insurance_benefits` como objetivo, evalúa cuál sería la RECM de un modelo de regresión lineal.

# Construye tu propia implementación de regresión lineal. Para ello, recuerda cómo está formulada la solución de la tarea de regresión lineal en términos de LA. Comprueba la RECM tanto para los datos originales como para los escalados. ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?
# 
# Denotemos- $X$: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades- $y$ — objetivo (un vector)- $\hat{y}$ — objetivo estimado (un vector)- $w$ — vector de pesos
# La tarea de regresión lineal en el lenguaje de las matrices puede formularse así:
# $$
# y = Xw
# $$
# 
# El objetivo de entrenamiento es entonces encontrar esa $w$ w que minimice la distancia L2 (ECM) entre $Xw$ y $y$:
# 
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
# 
# Parece que hay una solución analítica para lo anteriormente expuesto:
# $$
# w = (X^T X)^{-1} X^T y
# $$
# 
# La fórmula anterior puede servir para encontrar los pesos $w$ y estos últimos pueden utilizarse para calcular los valores predichos
# $$
# \hat{y} = X_{val}w
# $$

# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

# Paso 1: Definir la Clase MyLinearRegression
# El pre-código sugiere construir una implementación personalizada de un modelo de regresión lineal. Este modelo se basará en la fórmula matemática para obtener los pesos 𝑤 y realizar predicciones.

# In[68]:


class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None    # Inicializa los pesos como None
    
    def fit(self, X, y):
         # Añadir una columna de unos a X para representar el término independiente (bias)
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
         # Calcular los pesos utilizando la fórmula: w = (X^T X)^-1 X^T y
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y

    def predict(self, X):
        # Añadir una columna de unos a X para representar el término independiente (bias)
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        # Realizar la predicción utilizando los pesos calculados
        y_pred =  X2 @ self.weights
        
        return y_pred


# Paso 2: Definir la Función de Evaluación eval_regressor
# Vamos a definir la función que evaluará el rendimiento del modelo utilizando la métrica RECM (RMSE) y el coeficiente de determinación R2.

# In[69]:


def eval_regressor(y_true, y_pred):
    # Calcular el RMSE
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    # Calcular el R2
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    


# Paso 3: Entrenar y Evaluar el Modelo de Regresión Lineal
# Ahora, entrenaremos el modelo de regresión lineal personalizado (MyLinearRegression) utilizando los datos de entrenamiento, y luego evaluaremos su rendimiento en los datos de prueba.

# In[70]:


# Paso 3: Preparar los datos y entrenar el modelo
X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

# Dividir los datos en entrenamiento y prueba (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Crear una instancia del modelo
lr = MyLinearRegression()

# Entrenar el modelo en los datos de entrenamiento
lr.fit(X_train, y_train)
print(lr.weights)

# Realizar predicciones en los datos de prueba
y_test_pred = lr.predict(X_test)
# Evaluar el modelo en los datos de prueba
eval_regressor(y_test, y_test_pred)


# Paso 4: Evaluar el Modelo con Datos Escalados
# También es útil escalar los datos para ver si esto afecta la precisión del modelo. Vamos a utilizar el MaxAbsScaler como en las tareas anteriores y repetir el proceso.

# In[71]:


# Escalar los datos
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear una instancia del modelo
lr_scaled = MyLinearRegression()

# Entrenar el modelo en los datos de entrenamiento escalados
lr_scaled.fit(X_train_scaled, y_train)

# Realizar predicciones en los datos de prueba escalados
y_test_pred_scaled = lr_scaled.predict(X_test_scaled)

# Evaluar el modelo en los datos de prueba escalados
eval_regressor(y_test, y_test_pred_scaled)


# El modelo de regresión lineal personalizado ha demostrado ser eficaz para predecir el número de prestaciones de seguro que un nuevo cliente podría recibir, con un buen nivel de precisión (RMSE bajo y R2 aceptable). Además, el hecho de que los resultados sean consistentes tanto para los datos originales como para los escalados sugiere que el modelo es robusto y bien ajustado.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy bien! Como pudiste notar usar los datos originales o los escalados no afectan los resultados del modelo de regresión lineal debido a que la relación lineal entre las variables no cambia, simplemente se reescalan. 
#     
# </div>

# # Tarea 4. Ofuscar datos

# Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz $X$) por una matriz invertible $P$. 
# 
# $$
# X' = X \times P
# $$
# 
# Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto, la propiedad de invertibilidad es importante aquí, así que asegúrate de que $P$ sea realmente invertible.
# 
# Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla de multiplicación de matrices y su implementación con NumPy.

# Paso 1: Ofuscar los Datos con una Matriz Invertible
# Primero, necesitamos crear una matriz aleatoria 𝑃 y comprobar que sea invertible. Luego, multiplicaremos la matriz 
# 𝑋 (características numéricas de los datos) por 𝑃 para obtener los datos ofuscados X.
# 

# In[72]:


# Seleccionar las columnas con información personal
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[73]:


X = df_pn.to_numpy()


# Generar una matriz aleatoria $P$.

# In[74]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))


# Comprobar que la matriz P sea invertible

# In[75]:


if np.linalg.det(P) != 0:
    print("La matriz P es invertible.")
else:
    raise ValueError("La matriz P no es invertible. Intente generar otra matriz.")

# Ofuscar los datos
X_obfuscated = X @ P

# Mostrar algunos ejemplos de los datos originales y ofuscados
print("Datos originales:\n", X[:5])
print("Datos ofuscados:\n", X_obfuscated[:5])


# ¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?

# Aquí, el objetivo es demostrar que si conocemos la matriz de transformación 𝑃, podemos recuperar los datos originales aplicando la operación inversa. Esto se puede hacer invirtiendo la matriz 𝑃 y multiplicando los datos transformados por la inversa de 𝑃.

# ¿Puedes recuperar los datos originales de $X'$ si conoces $P$? Intenta comprobarlo a través de los cálculos moviendo $P$ del lado derecho de la fórmula anterior al izquierdo. En este caso las reglas de la multiplicación matricial son realmente útiles

# Paso 2: Intentar Recuperar los Datos Originales
# Para comprobar que la transformación es reversible, vamos a intentar recuperar los datos originales utilizando la matriz 
# 𝑃−1

# In[76]:


# Recuperar los datos originales
P_inv = np.linalg.inv(P)
X_recovered = X_obfuscated @ P_inv


# Muestra los tres casos para algunos clientes- Datos originales
# - El que está transformado- El que está invertido (recuperado)

# In[77]:


# Mostrar algunos ejemplos de los datos recuperados
print("\nDatos recuperados:\n", X_recovered[:5])

# Comparar los datos recuperados con los originales
print("\nDatos originales:\n", X[:5])  # Muestra los primeros 5 registros originales

print("\nDatos transformados:\n", X_obfuscated[:5])  # Muestra los primeros 5 registros transformados

print("\nDatos recuperados:\n", X_recovered[:5])  # Muestra los primeros 5 registros recuperados


# Seguramente puedes ver que algunos valores no son exactamente iguales a los de los datos originales. ¿Cuál podría ser la razón de ello?

# En la práctica, la inversión de matrices y la multiplicación de matrices pueden introducir pequeños errores debido a la naturaleza de los cálculos en coma flotante. Esto significa que los valores recuperados podrían no ser exactamente iguales a los originales, aunque deberían ser muy cercanos.
# 
# Si los valores recuperados no son exactamente iguales, esto se debe a la acumulación de errores numéricos en la operación de inversión y multiplicación de matrices.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelente! Lograste ofuscar los datos originales de forma adecuada y después recuperarlos a través de operaciones matriciales. Si bien hay una diferencia muy pequeña entre los datos orginales y los datos recuperados, se debe a los redondeos que se están realizando al hacer las operaciones.
# </div>

# ## Prueba de que la ofuscación de datos puede funcionar con regresión lineal

# En este proyecto la tarea de regresión se ha resuelto con la regresión lineal. Tu siguiente tarea es demostrar _analytically_ que el método de ofuscación no afectará a la regresión lineal en términos de valores predichos, es decir, que sus valores seguirán siendo los mismos. ¿Lo puedes creer? Pues no hace falta que lo creas, ¡tienes que que demostrarlo!

# Entonces, los datos están ofuscados y ahora tenemos $X \times P$ en lugar de tener solo $X$. En consecuencia, hay otros pesos $w_P$ como
# $$
# w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
# $$
# 
# ¿Cómo se relacionarían $w$ y $w_P$ si simplificáramos la fórmula de $w_P$ anterior? 
# 
# ¿Cuáles serían los valores predichos con $w_P$? 
# 
# ¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM?
# Revisa el Apéndice B Propiedades de las matrices al final del cuaderno. ¡Allí encontrarás fórmulas muy útiles!
# 
# No es necesario escribir código en esta sección, basta con una explicación analítica.

# **Respuesta**

# ¿Cómo se relacionarían  $𝑤$ y  $𝑤_𝑃$ si simplificáramos la fórmula de  $𝑤_𝑃$ anterior?
# 
# Los nuevos pesos $w_P$ están relacionados con los pesos originales $w$ a través de la inversa de la matriz de transformación 𝑃.
# 
# En concreto:  $W_P$=$P^{−1}$ $w$
# 
# ¿Cuáles serían los valores predichos con  $𝑤_𝑃$?
# 
# Los valores predichos $\hat{y}p$ utilizando los datos ofuscados serán exactamente los mismos que los valores predichos utilizando los datos originales $\hat{y}$ Esto demuestra que la ofuscación de datos no afecta la calidad de las predicciones en términos de la regresión lineal.
# $\hat{y}p$= $XI_w$ = Xw = $\hat{y}$
#  
# ¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM? Revisa el Apéndice B Propiedades de las matrices al final del cuaderno. ¡Allí encontrarás fórmulas muy útiles!
# 
# Dado que los valores predichos $\hat{y}p$ son idénticos a los valores predichos $\hat{y}$ cuando utilizamos la ofuscación, las métricas de evaluación como el RMSE (Root Mean Squared Error) y el $𝑅^2$ no se verán afectadas. Por lo tanto, la calidad del modelo de regresión lineal permanecerá inalterada.

# **Prueba analítica**

# Pregunta 1: ¿Cómo se relacionarían los pesos $𝑤$ y $𝑤𝑃$ si simplificáramos la fórmula de $𝑤𝑃$?
# 
# *Partimos de la fórmula general de los pesos en la regresión lineal:
# 
# w = ($X^T$ X)$^{-1}$ $X^T$y
# 
# *Con la matriz de características ofuscada $X_p$ = X x P los nuevos pesos $W_p$ se calculan como:
# 
# $w_P$ = [$(X_P)^T$ $X_P$]$^{-1}$ $(X_P)^T$ $y$  = [$P^T$ $X^T$ $XP$]$^{-1}$ $P^T$ $X^T$ $y$
# 
# *Aplicando las propiedades de las matrices, podemos simplificar esto:
# 
# $w_P$ = $P^{-1}$ ($X^T$ $X$)$^{-1}$ $X^T$ $y$ = $P^{-1}$ $w$
# 
# Conclusión: Los nuevos pesos $w_p$ están relacionados con los pesos originales $w$ a través de la matriz inversa de la transformación $𝑃$
# 
# Pregunta 2: ¿Cuáles serían los valores predichos con $w_p$ ?
# 
# *Los valores predichos en la regresión lineal se obtienen como:
# 
# $\hat{y}$ = Xw
# 
# *Para los datos ofuscados, se tiene:
# 
# $\hat{y}p$= $X_p$$w_p$= ( X x P ) ($P^{-1}$w)
# 
# *Simplificando:
# 
# $\hat{y}p$= $XI_w$ = Xw = $\hat{y}$
# 
# *Conclusión: Los valores predichos con $w_p$ serán idénticos a los valores predichos con 𝑤
# 
# Pregunta 3 : La ofuscación de datos mediante la multiplicación por una matriz invertible $𝑃$ no afecta la calidad de la regresión lineal, ni en términos de los valores predichos ni en términos de las métricas de evaluación como RMSE o $R^2$
# 

# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelente! Solamente verifica la manera en que muestras el apartado dado que parece existir un error en el código
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante.</b> <a class="tocSkip"></a>
# </div>
# 
# No entendi a que se refiere que hay error de codigo , no entiendo el comentario para corregirlo , por favor entregarme mas detalles al respecto porque es la escritura de algebra de formulas lo que esta aqui y no codigo , entonces ahi me perdi. Favor mas detalles para poder corregir la informacion. Saludos 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Hola, ya se puede ver correctamente el código. Puede ser un problema de la plataforma y en la iteración pasada me marcada error
# </div>

# ## Prueba de regresión lineal con ofuscación de datos

# Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida.
# Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una implementación de regresión lineal de scikit-learn o tu propia implementación.
# Ejecuta la regresión lineal para los datos originales y los ofuscados, compara los valores predichos y los valores de las métricas RMSE y $R^2$. ¿Hay alguna diferencia?

# **Procedimiento**
# 
# - Crea una matriz cuadrada $P$ de números aleatorios.- Comprueba que sea invertible. Si no lo es, repite el primer paso hasta obtener una matriz invertible.- <¡ tu comentario aquí !>
# - Utiliza $XP$ como la nueva matriz de características

# Paso 1: Crear una matriz cuadrada 𝑃 de números aleatorios e invertirla.

# In[78]:


# Ya se generó anteriormente una matriz P y se comprobó que es invertible.
# Continuamos utilizando esa misma matriz P y P_inv.


# Paso 2: Utilizar 𝑋×𝑃 como la nueva matriz de características y comparar los resultados.

# In[79]:


# Crear el modelo de regresión lineal utilizando los datos originales
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)

# RMSE y R2 para los datos originales
rmse_original = mean_squared_error(y_test, y_pred_original, squared=False)
r2_original = r2_score(y_test, y_pred_original)

# Crear el modelo de regresión lineal utilizando los datos ofuscados
X_train_obfuscated = X_train @ P
X_test_obfuscated = X_test @ P

model_obfuscated = LinearRegression()
model_obfuscated.fit(X_train_obfuscated, y_train)
y_pred_obfuscated = model_obfuscated.predict(X_test_obfuscated)

# RMSE y R2 para los datos ofuscados
rmse_obfuscated = mean_squared_error(y_test, y_pred_obfuscated, squared=False)
r2_obfuscated = r2_score(y_test, y_pred_obfuscated)

# Mostrar los resultados
print(f"RMSE (original): {rmse_original:.4f}")
print(f"R2 (original): {r2_original:.4f}")
print(f"RMSE (ofuscado): {rmse_obfuscated:.4f}")
print(f"R2 (ofuscado): {r2_obfuscated:.4f}")


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Como ya demostraste anteriormente no existe diferencias entre el RMSE y R2 usando los datos originales y los datos ofuscados.
# </div>

# # Conclusiones

# 1. Igualdad de Resultados Entre Datos Originales y Ofuscados:
# RMSE (Raíz del Error Cuadrático Medio): Tanto para los datos originales como para los datos ofuscados, la RMSE es exactamente la misma (0.3436). Esto significa que la precisión del modelo, en términos de error medio, no se ve afectada por la ofuscación de los datos.
# R² (Coeficiente de Determinación): Al igual que la RMSE, el R² es idéntico para ambos conjuntos de datos (0.4305). Esto indica que la capacidad del modelo para explicar la varianza en los datos objetivo es la misma, independientemente de si los datos han sido ofuscados o no.

# 2. Impacto de la Ofuscación en la Regresión Lineal:
# La ofuscación de datos mediante la multiplicación de las características por una matriz invertible no altera los resultados de la regresión lineal. Los valores predichos y las métricas de evaluación (RMSE y R²) permanecen inalterados.
# Esto se debe a que la multiplicación por una matriz invertible y su inversa es una transformación lineal que preserva la estructura de los datos en términos de relaciones lineales. La regresión lineal, siendo una técnica que busca relaciones lineales entre las características y el objetivo, es robusta frente a este tipo de transformaciones.

# 3. Implicaciones Prácticas:
# La ofuscación de datos es una técnica viable para proteger la información sensible sin comprometer la precisión y eficacia de los modelos de machine learning, específicamente en el contexto de la regresión lineal.
# Esta propiedad es especialmente valiosa en aplicaciones donde la privacidad y la seguridad de los datos son críticas, permitiendo que se mantenga la funcionalidad del modelo sin exponer información confidencial.
# En resumen, la prueba demuestra que la ofuscación de datos, cuando se realiza correctamente, no afecta negativamente el rendimiento de la regresión lineal, lo que la convierte en una técnica efectiva para proteger los datos en contextos donde la privacidad es fundamental.

# # Lista de control

# Escribe 'x' para verificar. Luego presiona Shift+Enter.

# - [x]  Jupyter Notebook está abierto
# - [ ]  El código no tiene errores- [ ]  Las celdas están ordenadas de acuerdo con la lógica y el orden de ejecución
# - [ ]  Se ha realizado la tarea 1
#     - [ ]  Está presente el procedimiento que puede devolver k clientes similares para un cliente determinado
#     - [ ]  Se probó el procedimiento para las cuatro combinaciones propuestas    - [ ]  Se respondieron las preguntas sobre la escala/distancia- [ ]  Se ha realizado la tarea 2
#     - [ ]  Se construyó y probó el modelo de clasificación aleatoria para todos los niveles de probabilidad    - [ ]  Se construyó y probó el modelo de clasificación kNN tanto para los datos originales como para los escalados. Se calculó la métrica F1.- [ ]  Se ha realizado la tarea 3
#     - [ ]  Se implementó la solución de regresión lineal mediante operaciones matriciales    - [ ]  Se calculó la RECM para la solución implementada- [ ]  Se ha realizado la tarea 4
#     - [ ]  Se ofuscaron los datos mediante una matriz aleatoria e invertible P    - [ ]  Se recuperaron los datos ofuscados y se han mostrado algunos ejemplos    - [ ]  Se proporcionó la prueba analítica de que la transformación no afecta a la RECM    - [ ]  Se proporcionó la prueba computacional de que la transformación no afecta a la RECM- [ ]  Se han sacado conclusiones

# # Apéndices
# 
# ## Apéndice A: Escribir fórmulas en los cuadernos de Jupyter

# Puedes escribir fórmulas en tu Jupyter Notebook utilizando un lenguaje de marcado proporcionado por un sistema de publicación de alta calidad llamado $\LaTeX$ (se pronuncia como "Lah-tech"). Las fórmulas se verán como las de los libros de texto.
# 
# Para incorporar una fórmula a un texto, pon el signo de dólar (\\$) antes y después del texto de la fórmula, por ejemplo: $\frac{1}{2} \times \frac{3}{2} = \frac{3}{4}$ or $y = x^2, x \ge 1$.
# 
# Si una fórmula debe estar en el mismo párrafo, pon el doble signo de dólar (\\$\\$) antes y después del texto de la fórmula, por ejemplo:
# $$
# \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
# $$
# 
# El lenguaje de marcado de [LaTeX](https://es.wikipedia.org/wiki/LaTeX) es muy popular entre las personas que utilizan fórmulas en sus artículos, libros y textos. Puede resultar complicado, pero sus fundamentos son sencillos. Consulta esta [ficha de ayuda](http://tug.ctan.org/info/undergradmath/undergradmath.pdf) (materiales en inglés) de dos páginas para aprender a componer las fórmulas más comunes.

# ## Apéndice B: Propiedades de las matrices

# Las matrices tienen muchas propiedades en cuanto al álgebra lineal. Aquí se enumeran algunas de ellas que pueden ayudarte a la hora de realizar la prueba analítica de este proyecto.

# <table>
# <tr>
# <td>Distributividad</td><td>$A(B+C)=AB+AC$</td>
# </tr>
# <tr>
# <td>No conmutatividad</td><td>$AB \neq BA$</td>
# </tr>
# <tr>
# <td>Propiedad asociativa de la multiplicación</td><td>$(AB)C = A(BC)$</td>
# </tr>
# <tr>
# <td>Propiedad de identidad multiplicativa</td><td>$IA = AI = A$</td>
# </tr>
# <tr>
# <td></td><td>$A^{-1}A = AA^{-1} = I$
# </td>
# </tr>    
# <tr>
# <td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
# </tr>    
# <tr>
# <td>Reversibilidad de la transposición de un producto de matrices,</td><td>$(AB)^T = B^TA^T$</td>
# </tr>    
# </table>

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Gran trabajo! Que sigas disfrutando los siguientes cursos!
# </div>
