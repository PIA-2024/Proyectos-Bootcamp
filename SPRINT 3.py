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

# # ¡Llena ese carrito!

# # Introducción
# 
# Instacart es una plataforma de entregas de comestibles donde la clientela puede registrar un pedido y hacer que se lo entreguen, similar a Uber Eats y Door Dash.
# El conjunto de datos que te hemos proporcionado tiene modificaciones del original. Redujimos el tamaño del conjunto para que tus cálculos se hicieran más rápido e introdujimos valores ausentes y duplicados. Tuvimos cuidado de conservar las distribuciones de los datos originales cuando hicimos los cambios.
# 
# Debes completar tres pasos. Para cada uno de ellos, escribe una breve introducción que refleje con claridad cómo pretendes resolver cada paso, y escribe párrafos explicatorios que justifiquen tus decisiones al tiempo que avanzas en tu solución.  También escribe una conclusión que resuma tus hallazgos y elecciones.
# 

# ## Diccionario de datos
# 
# Hay cinco tablas en el conjunto de datos, y tendrás que usarlas todas para hacer el preprocesamiento de datos y el análisis exploratorio de datos. A continuación se muestra un diccionario de datos que enumera las columnas de cada tabla y describe los datos que contienen.
# 
# - `instacart_orders.csv`: cada fila corresponde a un pedido en la aplicación Instacart.
#     - `'order_id'`: número de ID que identifica de manera única cada pedido.
#     - `'user_id'`: número de ID que identifica de manera única la cuenta de cada cliente.
#     - `'order_number'`: el número de veces que este cliente ha hecho un pedido.
#     - `'order_dow'`: día de la semana en que se hizo el pedido (0 si es domingo).
#     - `'order_hour_of_day'`: hora del día en que se hizo el pedido.
#     - `'days_since_prior_order'`: número de días transcurridos desde que este cliente hizo su pedido anterior.
# - `products.csv`: cada fila corresponde a un producto único que pueden comprar los clientes.
#     - `'product_id'`: número ID que identifica de manera única cada producto.
#     - `'product_name'`: nombre del producto.
#     - `'aisle_id'`: número ID que identifica de manera única cada categoría de pasillo de víveres.
#     - `'department_id'`: número ID que identifica de manera única cada departamento de víveres.
# - `order_products.csv`: cada fila corresponde a un artículo pedido en un pedido.
#     - `'order_id'`: número de ID que identifica de manera única cada pedido.
#     - `'product_id'`: número ID que identifica de manera única cada producto.
#     - `'add_to_cart_order'`: el orden secuencial en el que se añadió cada artículo en el carrito.
#     - `'reordered'`: 0 si el cliente nunca ha pedido este producto antes, 1 si lo ha pedido.
# - `aisles.csv`
#     - `'aisle_id'`: número ID que identifica de manera única cada categoría de pasillo de víveres.
#     - `'aisle'`: nombre del pasillo.
# - `departments.csv`
#     - `'department_id'`: número ID que identifica de manera única cada departamento de víveres.
#     - `'department'`: nombre del departamento.

# # Paso 1. Descripción de los datos
# 
# Lee los archivos de datos (`/datasets/instacart_orders.csv`, `/datasets/products.csv`, `/datasets/aisles.csv`, `/datasets/departments.csv` y `/datasets/order_products.csv`) con `pd.read_csv()` usando los parámetros adecuados para leer los datos correctamente. Verifica la información para cada DataFrame creado.
# 

# ## Plan de solución
# 
# Escribe aquí tu plan de solución para el Paso 1. Descripción de los datos.

# In[1]:


import pandas as pd # importar librerías


# <div class="alert alert-block alert-warning">
# <b>Comentario revisor</b> <a class="tocSkip"></a>
# 
# 
# Recomiendo importar y cargar la librería de matplolib.pyplot para el desarrollo de las diferentes gráficas que te van a ayudar al análisis de los datos. 
# </div>

# In[2]:


instacart_orders_df = pd.read_csv('/datasets/instacart_orders.csv', sep = ';') # cargamos y leemos  el conjunto de datos en los DataFrames.
products_df = pd.read_csv('/datasets/products.csv', sep = ';')                 # se carga cada archivo con una variable para cada uno.
aisles_df = pd.read_csv('/datasets/aisles.csv', sep = ';')
departments_df = pd.read_csv('/datasets/departments.csv', sep = ';')
order_products_df = pd.read_csv('/datasets/order_products.csv', sep = ';')


# In[3]:


# aqui vamos a Explorar la información de cada DataFrame con los metodos mas explicitos.
# Aca se entregaran las primeras filas, la información de las columnas, estadisticas descriptivas y el tamaño del DataFrame 
print("Instacart Orders DataFrame:")      # mostrar información del DataFrame         
print(instacart_orders_df.head(10))
print(instacart_orders_df.info())


# In[4]:


print("\nProducts DataFrame:")         # mostrar información del DataFrame
print(products_df.head(10))
print(products_df.info())


# In[5]:


print("\nAisles DataFrame:")           # mostrar información del DataFrame
print(aisles_df.head(10))
print(aisles_df.info())


# In[6]:


print("\nDepartments DataFrame:")      # mostrar información del DataFrame
print(departments_df.head(10))
print(departments_df.info())


# In[7]:


print("\nOrder Products DataFrame:")    # mostrar información del DataFrame
print(order_products_df.head(10))
print(order_products_df.info())



# ## Conclusiones
# 
# Escribe aquí tus conclusiones intermedias sobre el Paso 1. Descripción de los datos.
# 

# Lo que se puede observar a primera vista es que existe un error en como vienen los archivos CSV , debido a que cuando se llaman para ver la información, los datos estan separados por un (;) y no por (,) lo cual hace que se muestren los datos en una sola columna. Esto significa que todas las columnas del DataFrame se muestran en una sola columna, ademas todos los datos entregados los muestra de tipo object.

# Con esto identificado, se tuvo que corregir los errores que se veian en los archivos , partiendo con poner un separador sep= ';' Con esto se pudieron separar las columnas de los diferentes DataFrame, para entregar la información ordenada y mas visible.
# Se identifico que hay valores ausentes en la columna 'days_since_prior_order' del dataframe Instacart Orders,valores ausentes en la columna product_name'del dataframe Products.
# Se observo que en el dataframe de order product el metodo info() no muestra en su tabla los valos non-null, al irnos al archivo excel original podemos identificar que existen valores vacios en la columna add_to_cart_order, por lo cual hay alguna irregularidad que hay que evaluar.
# 

# # Paso 2. Preprocesamiento de los datos
# 
# Preprocesa los datos de la siguiente manera:
# 
# - Verifica y corrige los tipos de datos (por ejemplo, asegúrate de que las columnas de ID sean números enteros).
# - Identifica y completa los valores ausentes.
# - Identifica y elimina los valores duplicados.
# 
# Asegúrate de explicar qué tipos de valores ausentes y duplicados encontraste, cómo los completaste o eliminaste y por qué usaste esos métodos. ¿Por qué crees que estos valores ausentes y duplicados pueden haber estado presentes en el conjunto de datos?

# ## Plan de solución
# 
# Escribe aquí tu plan para el Paso 2. Preprocesamiento de los datos.

# Teniendo la información clara de los DataFrame , con las correcciones necesarias para poder trabajar con ellos , se procedera a verificar y corregir errores que puedan haber en los datos , ver y rellenar valores ausentes si correpsonde junto con eliminar valores duplicados con el fin de poder dejar los datos limpios.

# In[8]:


# 1. Verificación y corrección de tipos de datos
# Convertir las columnas de ID a números enteros si es necesario
instacart_orders_df['order_id'] = instacart_orders_df['order_id'].astype(int)
instacart_orders_df['user_id'] = instacart_orders_df['user_id'].astype(int)
products_df['product_id'] = products_df['product_id'].astype(int)
products_df['aisle_id'] = products_df['aisle_id'].astype(int)
aisles_df['aisle_id'] = aisles_df['aisle_id'].astype(int)
departments_df['department_id'] = departments_df['department_id'].astype(int)
order_products_df['order_id'] = order_products_df['order_id'].astype(int)
order_products_df['product_id'] = order_products_df['product_id'].astype(int) 

# 2. Identificación y completado de valores ausentes
# Hay valores ausentes en 3 DataFrame Instacart, Product y Order Product.
print('Instacart')
print(instacart_orders_df.isna().sum())
print('Products')
print(products_df.isna().sum())
print('Aisles')
print(aisles_df.isna().sum())
print('Departments')
print(departments_df.isna().sum())
print('Order Products')
print(order_products_df.isna().sum())


# In[ ]:





# ## Encuentra y elimina los valores duplicados (y describe cómo tomaste tus decisiones).

# ### `orders` data frame

# In[9]:


# Revisa si hay pedidos duplicados
print("Pedidos duplicados:")
print(instacart_orders_df.duplicated().value_counts())
print()
print("Pedidos duplicados:")
duplicated_instacart_orders= instacart_orders_df[instacart_orders_df.duplicated()]
print(duplicated_instacart_orders)


# ¿Tienes líneas duplicadas? Si sí, ¿qué tienen en común?

# In[10]:


# Basándote en tus hallazgos
#  Si hay valores duplicados , se muestran 15 pedidos duplicados en el DataFrame de Instacart
# Los pedidos duplicados tienen en comun que fueron duplicados el dia 3 (order_dow) y a las 2 (order_hour_of_day)

# Verifica todos los pedidos que se hicieron el miércoles a las 2:00 a.m.
wednesday_orders = instacart_orders_df[(instacart_orders_df['order_dow'] == 3) & (instacart_orders_df['order_hour_of_day'] == 2)]
print("Pedidos realizados el miércoles a las 2:00 a.m.:")
print(wednesday_orders['order_id'].count())


# ¿Qué sugiere este resultado?

# El resultado de 121 pedidos realizados el miércoles a las 2:00 a.m. sugiere que hubo duplicados en los registros de pedidos para ese día y hora específicos. Esto se puede interpretar como una anomalía en el registro de datos, ya que es poco probable que haya tantos pedidos exactamente a la misma hora en la vida real.
# Que el dia miercoles a las 2 hubo un error y se duplicaron 15 pedidos, y que de los 121 pedidos que se hicieron ese dia a esa hora en realidad son solo 106 pedidos descontando los 15 duplicados.
# 
# Este resultado sugiere que los 15 pedidos duplicados identificados son un error en los datos y no representan transacciones únicas. Considerando esta información podriamos decir que de los 121 pedidos 15 son duplicados por lo que corresponden unicamente 106 pedidos a ese dia y a esa hora especifica.
# 
# Para abordar este problema, sería recomienda eliminar los pedidos duplicados del conjunto de datos.

# In[11]:


# Elimina los pedidos duplicados
instacart_orders_df.drop_duplicates(inplace=True)


# In[12]:


# Vuelve a verificar si hay filas duplicadas
print("Filas duplicadas después de eliminarlas:")
print(instacart_orders_df.duplicated().sum())


# In[13]:


# Vuelve a verificar únicamente si hay IDs duplicados de pedidos
print("IDs duplicados de pedidos:")
print(instacart_orders_df['order_id'].duplicated().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos

# Después de eliminar los pedidos duplicados del DataFrame de Instacart Orders, se verificó que ya no había filas duplicadas, ya que el recuento de filas duplicadas devolvió 0. Además, al revisar específicamente si había IDs de pedidos duplicados, el resultado también fue 0, lo que indica que no quedaban IDs de pedidos duplicados después de la eliminación.
# 
# Esto confirma que los 15 pedidos duplicados identificados previamente fueron eliminados correctamente del conjunto de datos y que ese dia a esa hora en especifico solo hubieron 106 pedidos.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy buen trabajo!! Desarrollaste de manera excelente el análisis de duplicados y eliminaste esos casos. 
#     
# </div>

# ### `products` data frame

# In[14]:


# Revisa si hay productos duplicados
print("Productos duplicados:")
print(products_df.duplicated().sum())
print(products_df.duplicated().value_counts())


# In[15]:


# Revisa únicamente si hay ID de departamentos duplicados
print("IDs duplicados de departamentos:")
print(products_df['department_id'].duplicated().sum())


# In[16]:


# Revisa únicamente si hay nombres duplicados de productos (convierte los nombres a letras mayúsculas para compararlos mejor)
products_df['product_name'] = products_df['product_name'].str.upper()
print("Nombres duplicados de productos:", products_df['product_name'].duplicated().sum())


# In[17]:


# Revisa si hay nombres duplicados de productos no faltantes
print(products_df['product_name'].value_counts(dropna=False)) # aqui revisamos si hay duplicados de productos sin nombre incluyendo los valores NaN


# Describe brevemente tus hallazgos y lo que hiciste con ellos.

# No hay productos completamente duplicados en el conjunto de datos, ya que la suma de duplicados es cero y todos los valores únicos son 49694.
# Hay 49673 IDs de departamentos duplicados, lo que sugiere que la mayoría de los departamentos se repiten en el DataFrame.
# Hay 1361 nombres de productos duplicados después de convertirlos a letras mayúsculas para una comparación más precisa.
# Hay 1258 valores NaN en la columna de nombres de productos, lo que sugiere que hay productos sin nombre.
# Considerando que hay valores NaN ,se entiende de que los valores NaN se consideran como valores duplicados de acuerdo con esto podemos decir que solo 103 pedidos corresponden a productos duplicados que no son NaN.

# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy buen trabajo!! Desarrollaste de manera excelente el análisis de duplicados. Para complementar el análisis, qué podríamos decir de estos productos duplicados? 
#     
# </div>

# ### `departments` data frame

# In[18]:


# Revisa si hay filas totalmente duplicadas
print("Cantidad de Departamentos duplicados:")
print(departments_df.duplicated().sum())


# In[19]:


# Revisa únicamente si hay IDs duplicadas de productos
print("IDs duplicados de departamentos:")
print(departments_df['department_id'].duplicated().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos.

# En el DataFrame de departamentos, no se encontraron filas completamente duplicadas ni IDs duplicados de departamentos. 

# ### `aisles` data frame

# In[20]:


# Revisa si hay filas totalmente duplicadas
print("Cantidad de Pasillos duplicados:")
print(aisles_df.duplicated().sum())


# In[21]:


# Revisa únicamente si hay IDs duplicadas de productos
print("IDs duplicados de pasillos:") 
print(aisles_df['aisle_id'].duplicated().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos.

# En el DataFrame de pasillos, no se encontraron filas completamente duplicadas ni IDs duplicados de pasillos. Esto indica que cada fila representa un pasillo único y que cada ID de pasillo es único en el conjunto de datos. 

# ### `order_products` data frame

# In[22]:


# Revisa si hay filas totalmente duplicadas
print("Cantidad de Ordenes de Productos duplicados:")
print(order_products_df.duplicated().sum())


# In[23]:


# Vuelve a verificar si hay cualquier otro duplicado engañoso

print('Order Products')
print(order_products_df.isna().sum())
print()
print("IDs duplicados de pedidos:", order_products_df['order_id'].duplicated().sum())
print("IDs duplicados de productos:", order_products_df['product_id'].duplicated().sum())



# Describe brevemente tus hallazgos y lo que hiciste con ellos.

# En el dataframe de order_products nos entrega los resultados que no existen filas duplicadas, pero se pudo observar que existen valores faltantes en la columna add_to_cart_order. En este DataFrame , al llamar inicialmente al metodo info() se pudo observar una irregularidad , ya que no mencionaba los non-null. Aqui podemos revelar esta irregularidad.
# 
# Además, al verificar duplicados en las columnas order_id y product_id, se encontró que hay un gran número de IDs duplicados tanto para pedidos como para productos. Estos duplicados pueden ser legítimos, ya que un mismo producto puede estar presente en varios pedidos y un pedido puede contener múltiples productos. Por lo tanto, no se tomaron acciones para eliminar estos duplicados, ya que son una característica esperada de los datos de pedidos de productos.

# ## Encuentra y elimina los valores ausentes
# 
# Al trabajar con valores duplicados, pudimos observar que también nos falta investigar valores ausentes:
# 
# * La columna `'product_name'` de la tabla products.
# * La columna `'days_since_prior_order'` de la tabla orders.
# * La columna `'add_to_cart_order'` de la tabla order_productos.

# ### `products` data frame

# In[24]:


# Encuentra los valores ausentes en la columna 'product_name'
print("Valores ausentes en los Nombres de Productos")
print(products_df['product_name'].isnull().value_counts())


# Describe brevemente cuáles son tus hallazgos.

# Los hallazgos indican que hay 1258 valores ausentes en la columna 'product_name' del DataFrame de productos. Lo cual puede significar que los valores faltantes puede ser que no esten registrados o que hubo un error con los datos.

# In[25]:


#  ¿Todos los nombres de productos ausentes están relacionados con el pasillo con ID 100?
print("Productos relacionados con el pasillo 100")
print(products_df[products_df['aisle_id']==100].count())


# Describe brevemente cuáles son tus hallazgos.

# los 1258 productos con valores ausentes estan relacionados con el pasillo 100

# In[26]:


# ¿Todos los nombres de productos ausentes están relacionados con el departamento con ID 21?
print("Productos relacionados con el departamento ID 21")
print(products_df[products_df['department_id']==21].count())


# Describe brevemente cuáles son tus hallazgos.

# Los hallazgos muestran que todos los productos con nombres ausentes están relacionados con el departamento con ID 21. 

# In[27]:


# Usa las tablas department y aisle para revisar los datos del pasillo con ID 100 y el departamento con ID 21.
datos=products_df.loc[(products_df['aisle_id']==100) & (products_df['department_id']==21)]
print(datos)


# Describe brevemente cuáles son tus hallazgos.

# Los hallazgos muestran que todos los productos con nombres ausentes están asociados con el pasillo con ID 100 y el departamento con ID 21. Esto confirma que los productos con nombres ausentes se encuentran específicamente en este pasillo y departamento.

# In[28]:


# Completa los nombres de productos ausentes con 'Unknown'
products_df['product_name'].fillna('Unknown',inplace=True)
print(products_df.isna().sum())


# Describe brevemente tus hallazgos y lo que hiciste con ellos.

# Después de completar los nombres de productos ausentes con 'Unknown', se verificó que ya no hay valores ausentes en la columna 'product_name' del DataFrame de productos.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy buen trabajo!! Desarrollaste de manera excelente el análisis de valores faltantes y los llenaste con "unknown".
#     
# </div>

# ### `orders` data frame

# In[29]:


#Encuentra los valores ausentes
print('Instacart')
print(instacart_orders_df.isna().sum())


# In[30]:


# ¿Hay algún valor ausente que no sea el primer pedido del cliente?   #esta parte no la entendi la verdad
# Filtrar los pedidos que no son el primer pedido del cliente
no_primeros_pedidos = instacart_orders_df[instacart_orders_df['order_number'] > 1]

# Verificar si hay valores ausentes en esos pedidos
valores_ausentes_no_primeros = no_primeros_pedidos.isnull().sum()

# Imprimir los valores ausentes encontrados
print("Valores ausentes en pedidos que no son el primer pedido del cliente:")
print(valores_ausentes_no_primeros)


# Describe brevemente tus hallazgos y lo que hiciste con ellos.

# En el DataFrame de Instacart Orders, se encontraron valores ausentes únicamente en la columna 'days_since_prior_order', con un total de 28,817 valores faltantes. Luego, se realizó una verificación adicional para determinar si estos valores ausentes correspondían a pedidos que no eran el primer pedido del cliente. Para esto, se filtraron los pedidos donde el número de pedido era mayor que 1. Después de este filtrado, se verificó que no había valores ausentes en estos pedidos. Esto sugiere que los valores ausentes en la columna 'days_since_prior_order' corresponden únicamente al primer pedido de cada cliente.

# ### `order_products` data frame

# In[31]:


# Encuentra los valores ausentes
print('Order Products')
print(order_products_df.isna().sum())


# In[32]:


# ¿Cuáles son los valores mínimos y máximos en esta columna?
minimo = order_products_df['add_to_cart_order'].min()
maximo = order_products_df['add_to_cart_order'].max()
print('Los valores mínimo y máximo son: ',minimo  ,maximo)


# Describe brevemente cuáles son tus hallazgos.

# Se visualizo que en el dataframe de order product hay 836 valores ausentes , de los couales corresponden a la columna add_to_cart_order.  De esta misma tabla , que corresponde al orden secuencial en el que se añadió cada artículo en el carrito obtuvimos el valor minimo y maximo.

# In[33]:


# Guarda todas las IDs de pedidos que tengan un valor ausente en 'add_to_cart_order'
valor_ausente_order_products =order_products_df[order_products_df['add_to_cart_order'].isnull()]['order_id']
print(valor_ausente_order_products)


# In[34]:


# ¿Todos los pedidos con valores ausentes tienen más de 64 productos?
valor_64 =order_products_df[order_products_df['add_to_cart_order'].isnull()]
pedidos_valor_64 = valor_64['order_id'].isin(order_products_df.groupby('order_id')['product_id'].count())
print(pedidos_valor_64.value_counts())
# Agrupa todos los pedidos con datos ausentes por su ID de pedido.
datos_aus = order_products_df[order_products_df['add_to_cart_order'].isnull()]
print(datos_aus)
# Cuenta el número de 'product_id' en cada pedido y revisa el valor mínimo del conteo.
print(order_products_df['product_id'].value_counts().sum())


# Describe brevemente cuáles son tus hallazgos.

# Se encontraron 836 pedidos con valores ausentes en la columna 'add_to_cart_order'. Se identificó que no todos estos pedidos tenían más de 64 productos, lo que indica que el valor máximo de 64 no se puede utilizar para completar los valores ausentes de 'add_to_cart_order'. Se agruparon todos los pedidos con datos ausentes por su ID de pedido y se contó el número de 'product_id' en cada pedido. Se verificó que el valor mínimo del conteo de 'product_id' era mayor que 1, lo que confirma que los pedidos con valores ausentes en 'add_to_cart_order' contenían más de un producto. Esto sugiere que no hay un valor específico que pueda reemplazar los valores ausentes en 'add_to_cart_order'.

# In[35]:


# Remplaza los valores ausentes en la columna 'add_to_cart? con 999 y convierte la columna al tipo entero.
order_products_df['add_to_cart_order'] = order_products_df['add_to_cart_order'].fillna(999).astype(int)
print(order_products_df['add_to_cart_order'])


# Describe brevemente tus hallazgos y lo que hiciste con ellos.

# Con el codigo que se propuso se pudo distinguir que en la columna de add_to_cart_order existian valores que eran ausentes, se tuvieron que ver que existian valores NaN para poder seguir con el codigo.
# Encontré que la columna 'add_to_cart_order' del dataframe 'order_products_df' tenía valores ausentes. Para abordar esto, primero reemplacé los valores ausentes con el número 999 utilizando el método fillna(999). Luego, intenté convertir la columna al tipo de datos entero utilizando el método astype(int). 

# <div class="alert alert-block alert-warning">
# <b>Comentario revisor</b> <a class="tocSkip"></a>
# 
# 
# Muy buena conclusión de esta base. Pero qué podríamos decir de los pedidos que tienen más de 64 productos?
# </div>

# ## Conclusiones
# 
# Escribe aquí tus conclusiones intermedias sobre el Paso 2. Preprocesamiento de los datos
# 

# Se pudo identificar que habian dataframe que era necesario corregir, para poder trabajar con ellos de forma ordenada, ademas se pudo observar que habian distintas columnas con valores ausentes y tambien valores duplicados. Se tuvo que hacer modificaciones como eliminacion de filas duplicadas, rellenar valores ausentes para poder trabajar los dataframe de mejor forma.
# En resumen, fue crucial para realizar la limpieza y preparar los datos para análisis posteriores, asegurando que estén libres de duplicados y valores ausentes, y que tengan el formato correcto para su manipulación y análisis.

# # Paso 3. Análisis de los datos
# 
# Una vez los datos estén procesados y listos, haz el siguiente análisis:

# # [A] Fácil (deben completarse todos para aprobar)
# 
# 1. Verifica que los valores en las columnas `'order_hour_of_day'` y `'order_dow'` en la tabla orders sean razonables (es decir, `'order_hour_of_day'` oscile entre 0 y 23 y `'order_dow'` oscile entre 0 y 6).
# 2. Crea un gráfico que muestre el número de personas que hacen pedidos dependiendo de la hora del día.
# 3. Crea un gráfico que muestre qué día de la semana la gente hace sus compras.
# 4. Crea un gráfico que muestre el tiempo que la gente espera hasta hacer su siguiente pedido, y comenta sobre los valores mínimos y máximos.

# ### [A1] Verifica que los valores sean sensibles

# In[36]:


valid_order_hour = (instacart_orders_df['order_hour_of_day']>=0 ) & (instacart_orders_df['order_hour_of_day'] <= 23)
print(valid_order_hour.all())


# In[37]:


valid_order_dow =(instacart_orders_df['order_dow']>=0 ) & (instacart_orders_df['order_dow'] <= 6)
print(valid_order_dow.all())


# Escribe aquí tus conclusiones

# Los valores en las columnas 'order_hour_of_day' y 'order_dow' en la tabla de órdenes son razonables.
# Todos los valores en la columna 'order_hour_of_day' están dentro del rango de 0 a 23.
# Todos los valores en la columna 'order_dow' están dentro del rango de 0 a 6.

# ### [A2] Para cada hora del día, ¿cuántas personas hacen órdenes?

# In[38]:


# Crear un gráfico que muestre el número de personas que hacen pedidos dependiendo de la hora del día
import matplotlib.pyplot as plt  # Importar la biblioteca matplotlib.pyplot
plt.figure(figsize=(10, 6))
instacart_orders_df['order_hour_of_day'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Número de Personas que hacen pedidos por Hora del Día')
plt.xlabel('Hora del Día')
plt.ylabel('Número de Peronas')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Escribe aquí tus conclusiones

# Entre las 10 y 16:00 horas en donde mas cantidad de personas hacen ordenes.
# El gráfico muestra claramente cómo varía el número de personas que hacen pedidos a lo largo del día.
# Se observa un pico en la cantidad de personas que hacen pedidos durante las horas diurnas, con un descenso durante las primeras horas de la mañana y durante la noche.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy buen trabajo!! Buen uso de la gráfica de barras para mostrar la distribución por horas.
#     
# </div>

# ### [A3] ¿Qué día de la semana compran víveres las personas?

# In[39]:


# Crear un gráfico que muestre qué día de la semana la gente hace sus compras
plt.figure(figsize=(10, 6))
instacart_orders_df['order_dow'].value_counts().sort_index().plot(kind='bar', color='orange')
plt.title('Día de la Semana para Pedidos')
plt.xlabel('Día de la Semana (0: Domingo, 1: Lunes, ..., 6: Sábado)')
plt.ylabel('Número de Pedidos')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Escribe aquí tus conclusiones

# Los dias domingo y lunes son los dias que mas compran los clientes debido a la cantidad de pedidos.

# ### [A4] ¿Cuánto tiempo esperan las personas hasta hacer otro pedido? Comenta sobre los valores mínimos y máximos.

# In[40]:


# Crear un gráfico de líneas con marcadores que muestre el tiempo que la gente espera hasta hacer su siguiente pedido
plt.figure(figsize=(10, 6))
instacart_orders_df['days_since_prior_order'].value_counts().sort_index().plot(kind='line', color='green', marker='o', linestyle='-')
plt.title('Días Transcurridos hasta el Siguiente Pedido')
plt.xlabel('Días desde el Pedido Anterior')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Establecer los ticks del eje x para que muestren valores de uno en uno
min_dias = int(instacart_orders_df['days_since_prior_order'].min())
max_dias = int(instacart_orders_df['days_since_prior_order'].max())
ticks = [i for i in range(min_dias, max_dias + 1)]
plt.xticks(ticks)

plt.show()
# Comentar sobre los valores mínimos y máximos del tiempo que la gente espera hasta hacer su siguiente pedido
min_dias = instacart_orders_df['days_since_prior_order'].min()
max_dias = instacart_orders_df['days_since_prior_order'].max()
print(f"El valor mínimo de días transcurridos hasta el siguiente pedido es: {min_dias}")
print(f"El valor máximo de días transcurridos hasta el siguiente pedido es: {max_dias}")


# Escribe aquí tus conclusiones

# La mayoría de las personas parece hacer pedidos con una frecuencia de entre 7 y 14 días, El valor 30 en el grafico tiene la mayor altura pero considero que se refiere al re inicio del mes.
# En este caso, el valor mínimo es 0 días, lo que indica que algunos usuarios hacen pedidos diariamente, mientras que el valor máximo es 30 días, lo que sugiere que otros usuarios esperan hasta un mes entre pedidos.

# # [B] Intermedio (deben completarse todos para aprobar)
# 
# 1. ¿Existe alguna diferencia entre las distribuciones `'order_hour_of_day'` de los miércoles y los sábados? Traza gráficos de barra de `'order_hour_of_day'` para ambos días en la misma figura y describe las diferencias que observes.
# 2. Grafica la distribución para el número de órdenes que hacen los clientes (es decir, cuántos clientes hicieron solo 1 pedido, cuántos hicieron 2, cuántos 3, y así sucesivamente...).
# 3. ¿Cuáles son los 20 principales productos que se piden con más frecuencia (muestra su identificación y nombre)?

# ### [B1] Diferencia entre miércoles y sábados para  `'order_hour_of_day'`. Traza gráficos de barra para los dos días y describe las diferencias que veas.

# In[41]:


# Filtrar pedidos para los miércoles y los sábados
wednesday_orders = instacart_orders_df[instacart_orders_df['order_dow'] == 3]
saturday_orders = instacart_orders_df[instacart_orders_df['order_dow'] == 6]

# Crear una tabla pivote para contar el número de pedidos por hora del día para ambos días
pivot_table = pd.pivot_table(instacart_orders_df.query("(order_dow == 3) or (order_dow == 6)"),
                             index="order_hour_of_day",
                             columns="order_dow",
                             values="order_id",
                             aggfunc="count")

# Graficar las barras para ambos días en el mismo gráfico
pivot_table.plot(kind="bar", figsize=(12, 6), ec="black")

# Añadir etiquetas y título al gráfico
plt.title("Distribución de Pedidos por Hora del Día (Miércoles vs. Sábado)")
plt.xlabel("Hora del Día")
plt.ylabel("Número de Pedidos")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir leyenda
plt.legend(["Miércoles", "Sábado"])

plt.show()


# Escribe aquí tus conclusiones

# Observamos que el número total de pedidos puede ser mayor los sábados en comparación con los miércoles, ya que las barras de los sábados son generalmente más altas que las de los miércoles en la mayoría de las horas del día.
# Horas pico: Podemos observar que los miércoles tienden a tener un pico de pedidos alrededor de las 10:00 AM y otro pico menor alrededor de las 15:00 PM. Mientras tanto, los sábados muestran un pico más marcado alrededor de las 10:00 AM y un segundo pico significativo alrededor de las 14:00 PM. Esto sugiere que las personas tienden a hacer sus compras más temprano los sábados en comparación con los miércoles.

# ### [B2] ¿Cuál es la distribución para el número de pedidos por cliente?

# In[42]:


# Calcular el número de pedidos por cliente
orders_per_customer = instacart_orders_df['user_id'].value_counts()
# Calcular la frecuencia de cada cantidad de pedidos
order_counts = orders_per_customer.value_counts().sort_index()


# In[43]:


# Graficar la distribución
plt.figure(figsize=(10, 6))
order_counts.plot(kind='bar', color='skyblue')
plt.title('Distribución del Número de Pedidos por Cliente')
plt.xlabel('Número de Pedidos')
plt.ylabel('Número de Clientes')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Calcular estadísticas
min_orders = order_counts.index.min()
max_orders = order_counts.index.max()
mode_orders = order_counts.idxmax()
median_orders = orders_per_customer.median()
mean_orders = orders_per_customer.mean()

# Imprimir estadísticas
print(f'Mínimo: {min_orders}')
print(f'Máximo: {max_orders}')
print(f'Moda: {mode_orders}')
print(f'Mediana: {median_orders}')
print(f'promedio: {mean_orders}')


# Escribe aquí tus conclusiones

# 
# El gráfico muestra la distribución del número de pedidos por cliente. Podemos observar que la mayoría de los clientes realizaron solo 1 pedido, seguido por un número considerable de clientes que realizaron 2 pedidos. A medida que aumenta el número de pedidos, la cantidad de clientes que realizaron ese número de pedidos disminuye gradualmente, lo que es esperado ya que es menos común que los clientes realicen una gran cantidad de pedidos.
# 
# Las estadísticas resaltan aún más estas observaciones:
# 
# El valor mínimo es 1, lo que indica que al menos un cliente realizó solo un pedido.
# El valor máximo es 28, lo que sugiere que el cliente con más pedidos realizó un total de 28.
# La moda es 1, lo que significa que el número más común de pedidos por cliente es 1.
# La mediana es 2.0, lo que sugiere que la mitad de los clientes realizaron 1 o 2 pedidos.
# El promedio es aproximadamente 3.04, lo que indica que, en promedio, cada cliente realizó alrededor de 3 pedidos.

# ### [B3] ¿Cuáles son los 20 productos más populares (muestra su ID y nombre)?

# In[44]:


# Primero, necesitamos combinar las tablas order_products_df y products_df
merged_df = pd.merge(order_products_df, products_df, on='product_id')
print(merged_df.head())


# In[45]:


# Luego, contamos la frecuencia de cada producto
product_counts = merged_df['product_name'].value_counts().head(20)

print(product_counts)


# In[46]:


# Mostramos los 20 productos más populares con su identificación y nombre
top_20_products = products_df[products_df['product_name'].isin(product_counts.index)]
top_20_products = top_20_products[['product_id', 'product_name']].drop_duplicates()

print(top_20_products)


# In[47]:


import matplotlib.pyplot as plt

# Graficar la distribución de los 20 productos más populares
plt.figure(figsize=(12, 6))
product_counts.plot(kind='bar', color='skyblue')
plt.title('Distribución de los 20 productos más populares')
plt.xlabel('Producto')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x para una mejor legibilidad
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Escribe aquí tus conclusiones

# Primero se combinan las tablas order_products_df y products_df utilizando la columna product_id como clave de unión. Luego, se cuentan las frecuencias de cada producto y se muestran los 20 productos más frecuentes junto con sus identificaciones y nombres.
# En el grafico se pueden ver los productos con mayor frecuencia de compra, 'BANANA' y 'BAG OF ORGANIC BANANA'

# # [C] Difícil (deben completarse todos para aprobar)
# 
# 1. ¿Cuántos artículos suelen comprar las personas en un pedido? ¿Cómo es la distribución?
# 2. ¿Cuáles son los 20 principales artículos que vuelven a pedirse con mayor frecuencia (muestra sus nombres e IDs de los productos)?
# 3. Para cada producto, ¿cuál es la tasa de repetición del pedido (número de repeticiones de pedido/total de pedidos?
# 4. Para cada cliente, ¿qué proporción de los productos que pidió ya los había pedido? Calcula la tasa de repetición de pedido para cada usuario en lugar de para cada producto.
# 5. ¿Cuáles son los 20 principales artículos que la gente pone primero en sus carritos (muestra las IDs de los productos, sus nombres, y el número de veces en que fueron el primer artículo en añadirse al carrito)?

# ### [C1] ¿Cuántos artículos compran normalmente las personas en un pedido? ¿Cómo es la distribución?

# In[48]:


# Calcular el número de artículos por pedido
items_per_order = merged_df.groupby('order_id')['product_id'].count()
print(items_per_order)


# In[49]:


# Calcular estadísticas descriptivas
mean_items_per_order = items_per_order.mean()
median_items_per_order = items_per_order.median()
min_items_per_order = items_per_order.min()
max_items_per_order = items_per_order.max()
print("Media:", mean_items_per_order)
print("Mediana:", median_items_per_order)
print("Mínimo:", min_items_per_order)
print("Máximo:", max_items_per_order)


# In[50]:


# Graficar la distribución
plt.figure(figsize=(10, 6))
items_per_order.hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Distribución del Número de Artículos por Pedido')
plt.xlabel('Número de Artículos por Pedido')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Escribe aquí tus conclusiones

# De acuerdo a cuántos artículos compran normalmente las personas en un pedido:
# La media (promedio) del número de artículos por pedido es aproximadamente 10.1.
# La mediana del número de artículos por pedido es 8.
# El número mínimo de artículos en un pedido es 1, lo que indica que hay pedidos que consisten en un solo artículo.
# El número máximo de artículos en un pedido es 127, lo que sugiere que hay algunos pedidos muy grandes con una cantidad significativa de artículos.
# Por lo tanto, la distribución del número de artículos por pedido muestra que la mayoría de los pedidos contienen un número relativamente bajo de artículos, mientras que un número mucho menor de pedidos contienen un número mayor de artículos.

# ### [C2] ¿Cuáles son los 20 principales artículos que vuelven a pedirse con mayor frecuencia (muestra sus nombres e IDs de los productos)?

# In[51]:


# Filtrar solo los productos que se han pedido más de una vez
repeated_products = merged_df['product_id'].value_counts()[merged_df['product_id'].value_counts() > 1]


# In[52]:


# Contar la frecuencia de estos productos en los pedidos
top_repeated_products = merged_df[merged_df['product_id'].isin(repeated_products.index)]


# In[53]:


# Seleccionar los 20 productos con la frecuencia más alta
top_20_repeated_products = top_repeated_products['product_id'].value_counts().head(20)
# Obtener los nombres de los productos
product_names = products_df.set_index('product_id')['product_name']
# Mostrar los nombres e IDs de los 20 principales productos que se vuelven a pedir con mayor frecuencia
top_20_products_info = product_names[top_20_repeated_products.index]
print(top_20_products_info)


# Escribe aquí tus conclusiones

# Podemos concluir que los 20 principales productos que se vuelven a pedir con mayor frecuencia son productos populares entre los clientes 

# ### [C3] Para cada producto, ¿cuál es la proporción de las veces que se pide y que se vuelve a pedir?

# In[62]:


# Calcular la cantidad de veces que cada producto se pide y se vuelve a pedir
relacion = merged_df.groupby('product_id').agg(veces_pedido=('order_id', 'size'),  # Total de veces que se ha pedido el producto
veces_repetido=('reordered', 'sum')  )     # Total de veces que se ha vuelto a pedir el producto

relacion['tasa'] = relacion['veces_repetido'] / relacion['veces_pedido']

print(relacion.head(15))


# Escribe aquí tus conclusiones

# Tasa alta: Los productos con una proporción cercana a 1 (por ejemplo, 0.9 o superior) productos populares que los clientes compran con frecuencia y vuelven a pedir.
# 
# Tasa baja: Los productos con una proporción cercana a 0 (por ejemplo, 0.1 o inferior) indican que los clientes rara vez vuelven a pedir esos productos después del primer pedido.
# 
# Tasa dintermedia: Los productos con una proporción en algún lugar entre 0 y 1 muestran una tasa de repetición moderada. 

# ### [C4] Para cada cliente, ¿qué proporción de sus productos ya los había pedido?

# In[63]:


cliente = pd.merge(instacart_orders_df, order_products_df, on='order_id', how='left')
repetidos = cliente.groupby('user_id').agg(
    pedidos=('order_id', 'nunique'),  # Total de pedidos realizados por el cliente
    repetido=('reordered', 'sum')      # Total de productos reordenados por el cliente
)
repetidos['repetido'] = repetidos['repetido'] / repetidos['pedidos']
repetidos = repetidos.reset_index()

print(repetidos)


# Escribe aquí tus conclusiones

# Esta información calculada nos entrega información sobre los clientes y su tendencia a repetir la compra de ciertos productos.
# La proporción de productos repetidos por cliente varía considerablemente, lo que sugiere que algunos clientes tienden a comprar una selección más estable de productos,mientras que otros son más propensos a probar nuevos productos en cada pedido.

# ### [C5] ¿Cuáles son los 20 principales artículos que las personas ponen primero en sus carritos?

# In[68]:


# Combina las tablas order_products y products
merged_df = pd.merge(order_products_df, products_df, on='product_id')

# Filtra los artículos que se pusieron primero en el carrito (add_to_cart_order == 1)
first_in_cart = merged_df[merged_df['add_to_cart_order'] == 1]

# Agrupa los datos por product_name y product_id y cuenta la frecuencia de cada artículo
top_first_in_cart = first_in_cart.groupby(['product_id', 'product_name']).size().reset_index(name='frequency')

# Ordena los resultados para encontrar los 20 principales artículos
top_20_first_in_cart = top_first_in_cart.sort_values(by='frequency', ascending=False).head(20)

print(top_20_first_in_cart)


# Escribe aquí tus conclusiones

# Aca podemos concluir que los clientes ingresan diferentes tipos de productos al carrito y que tienen diversas opciones de eleccion.

# ### Conclusion general del proyecto:

# De acuerdo a todo el trabajo realizado se pudieron observar:
# Patrones de compra por día y hora:  los patrones de compra varían según el día de la semana y la hora del día.
# 
# Distribución del número de pedidos por cliente: Hemos encontrado que la mayoría de los clientes realizan uno o dos pedidos, con una mediana de alrededor de 2 pedidos por cliente.
# 
# Productos más populares: Identificamos los productos más populares en la plataforma.
# 
# Tasa de repetición de pedidos: Analizamos la proporción de productos que se vuelven a pedir en comparación con el total de pedidos.
# 
# Productos agregados primero al carrito: Investigamos los productos que los clientes suelen agregar primero a sus carritos, lo que puede indicar hábitos de compra.
# 
# En resumen, este proyecto nos ha permitido obtener una visión completa del comportamiento de compra de los clientes en Instacart, lo que puede ser invaluable para la toma de decisiones comerciales, como estrategias de marketing, gestión de inventario y mejoras en la experiencia del cliente.
