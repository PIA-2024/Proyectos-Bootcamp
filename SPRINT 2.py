#!/usr/bin/env python
# coding: utf-8

# # ¡Hola Pia!
# 
# Mi nombre es Ezequiel Ferrario, soy code reviewer en Tripleten y tengo el agrado de revisar el proyecto que entregaste.
# 
# Para simular la dinámica de un ambiente de trabajo, si veo algún error, en primer instancia solo los señalaré, dándote la oportunidad de encontrarlos y corregirlos por tu cuenta. En un trabajo real, el líder de tu equipo hará una dinámica similar. En caso de que no puedas resolver la tarea, te daré una información más precisa en la próxima revisión.
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres**.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta. Se aceptan uno o dos comentarios de este tipo en el borrador, pero si hay más, deberá hacer las correcciones. Es como una tarea de prueba al solicitar un trabajo: muchos pequeños errores pueden hacer que un candidato sea rechazado.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# 
# Hola, muchas gracias por tus comentarios y la revisión.
# </div>
# 
# ¡Empecemos!

# <div class="alert alert-block alert-danger">
# <b>Comentario general #1</b> <a class="tocSkip"></a>
# 
# Pia, hiciste un muy buen trabajo. Donde practicamente no tenes correcciones realizadas.
# 
# Entendiste a nivel codigo que se requeria y tus analisis son muy buenos. Destaco ademas cada comentario que le haces a la celda para que quede mas claro.
# 
# A su vez el trabajo esta completo y solo resta un detalle para corregir.
# 
# Espero tus correcciones, saludos.</div>

# # Déjame escuchar la música

# # Contenido <a id='back'></a>
# 
# * [Introducción](#intro)
# * [Etapa 1. Descripción de los datos](#data_review)
#     * [Conclusiones](#data_review_conclusions)
# * [Etapa 2. Preprocesamiento de datos](#data_preprocessing)
#     * [2.1 Estilo del encabezado](#header_style)
#     * [2.2 Valores ausentes](#missing_values)
#     * [2.3 Duplicados](#duplicates)
#     * [2.4 Conclusiones](#data_preprocessing_conclusions)
# * [Etapa 3. Prueba de hipótesis](#hypothesis)
#     * [3.1 Hipótesis 1: actividad de los usuarios y las usuarias en las dos ciudades](#activity)
# * [Conclusiones](#end)

# <div class="alert alert-block alert-warning">
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy bien la tabla de contenidos pero debe estar linkeada a las secciones (al clickear debe llevarnos a esa seccion) de esta menera es mas facil desplazarse.
# 
# Como consejo, si realizas bien todas las secciones (con su respectivo #) podes generarlo automáticamente desde jupyter lab. Para hacerlo, en la pestaña de herramientas de jupyter lab clickeas en el **botón de los puntos y barras**  (Table of contents) te generara automáticamente una tabla de contenidos linkeable y estética. A la **derecha** del botón "Validate"
# </div>

# La verdad que esta parte no la entendi , uno porque la tabla de contenidos ya se encuentra creada al iniciar el programa , no es algo que uno haga. Por otra parte tampoco se mencionaba el linkearla , por lo que no lo hice , pero de acuerdo a su recomendación y pasos para realizarlo , de igual forma no me funciona; ingrese a tabla de contenidos y la tabla (como ya estaba hecha) estaba generada ya , puse validar pero aun asi no se linkean. Esto no se como realizarlo la verdad. 

# ## Introducción <a id='intro'></a>
# Como analista de datos, tu trabajo consiste en analizar datos para extraer información valiosa y tomar decisiones basadas en ellos. Esto implica diferentes etapas, como la descripción general de los datos, el preprocesamiento y la prueba de hipótesis.
# 
# Siempre que investigamos, necesitamos formular hipótesis que después podamos probar. A veces aceptamos estas hipótesis; otras veces, las rechazamos. Para tomar las decisiones correctas, una empresa debe ser capaz de entender si está haciendo las suposiciones correctas.
# 
# En este proyecto, compararás las preferencias musicales de las ciudades de Springfield y Shelbyville. Estudiarás datos reales de transmisión de música online para probar la hipótesis a continuación y comparar el comportamiento de los usuarios y las usuarias de estas dos ciudades.
# 
# ### Objetivo:
# Prueba la hipótesis:
# 1. La actividad de los usuarios y las usuarias difiere según el día de la semana y dependiendo de la ciudad.
# 
# 
# ### Etapas
# Los datos del comportamiento del usuario se almacenan en el archivo `/datasets/music_project_en.csv`. No hay ninguna información sobre la calidad de los datos, así que necesitarás examinarlos antes de probar la hipótesis.
# 
# Primero, evaluarás la calidad de los datos y verás si los problemas son significativos. Entonces, durante el preprocesamiento de datos, tomarás en cuenta los problemas más críticos.
# 
# Tu proyecto consistirá en tres etapas:
#  1. Descripción de los datos.
#  2. Preprocesamiento de datos.
#  3. Prueba de hipótesis.
# 
# 
# 
# 
# 
# 
# 

# [Volver a Contenidos](#back)

# ## Etapa 1. Descripción de los datos <a id='data_review'></a>
# 
# Abre los datos y examínalos.

# Necesitarás `pandas`, así que impórtalo.

# In[1]:


import pandas as pd # Importar pandas


# Lee el archivo `music_project_en.csv` de la carpeta `/datasets/` y guárdalo en la variable `df`:

# In[2]:


df = pd.read_csv('/datasets/music_project_en.csv')  # Leemos el archivo csv y lo guardamos en df para poder trabajarlo.


# Muestra las 10 primeras filas de la tabla:

# In[3]:


df.head(10)  # Obtener las 10 primeras filas de la tabla df. 
                     #Aqui el metodo nos entrega cuales son los datos que hay en las filas y ademas cuantas y cuales columnas son.


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Esta bien. Pero en este caso remplaza el print() y utiliza el metodo sin nada (llamalo solo). Es decir llamas a la variable y al metodo correspondiente y ejecutas.
# 
# - *El ejemplo seria: nombre_df.metodo(10)*
#     
# Esto hara que sea mas legible y estetico.</div>

# Eliminado el print

# Obtén la información general sobre la tabla con un comando. Conoces el método que muestra la información general que necesitamos.

# In[4]:


print(df.info())  # Obtener la información general sobre nuestros datos
                  # Aqui la información nos dice que 7 columnas de tipo object, donde tenemos valores no nulos, otros nulos y de tipo objeto


# Estas son nuestras observaciones sobre la tabla. Contiene siete columnas. Almacenan los mismos tipos de datos: `object`.
# 
# Según la documentación:
# - `' userID'`: identificador del usuario o la usuaria;
# - `'Track'`: título de la canción;
# - `'artist'`: nombre del artista;
# - `'genre'`: género de la pista;
# - `'City'`: ciudad del usuario o la usuaria;
# - `'time'`: la hora exacta en la que se reprodujo la canción;
# - `'Day'`: día de la semana.
# 
# Podemos ver tres problemas con el estilo en los encabezados de la tabla:
# 1. Algunos encabezados están en mayúsculas, otros en minúsculas.
# 2. Hay espacios en algunos encabezados.
# 3. `Detecta el tercer problema por tu cuenta y descríbelo aquí`.
# 
# Respuesta: El tercer error que veo corresponde userID lo cual tiene dos palabras sin separación , añadiria un _ para separar las dos palabras  'user_id'.
# 

# ### Escribe observaciones de tu parte. Estas son algunas de las preguntas que pueden ser útiles: <a id='data_review_conclusions'></a>
# 
# `1.   ¿Qué tipo de datos tenemos a nuestra disposición en las filas? ¿Y cómo podemos entender lo que almacenan las columnas?`
# 
# `2.   ¿Hay suficientes datos para proporcionar respuestas a nuestra hipótesis o necesitamos más información?`
# 
# `3.   ¿Notaste algún problema en los datos, como valores ausentes, duplicados o tipos de datos incorrectos?`

# 2.1 Respuestas
# 1. Los datos que que hay en las filas corresponden a 65078 datos que son object. Los datos que se almacen corresponden a la información por usuario de acuerdo a la ciudad que vive , el dia y hora en la cual escucho una determinada canción , junto al nombre de esta y el artista. El desglose de la información se puede ver en las columnas que entrega el dataframe.
# 2. No hay los suficientes datos para entregar una repsuesta a la hipótesis , se necesita trabajar con los datos para entregar una información respecto a la hipótesis. Se puede ver que hay información diferente a simple vista , pero visualemnte no podemos sacar un calculo para ver efectivamente si existe una referencia significativa para responder.
# 3. Hay valores en la tabla de datos que es necesario modificar , en la columna artist , track y genre , se pueden ver valores unknown de los cuales necesitamos aplicar metodos para corregirlos y poder trabajar con la información. Quizas artist y track no es necesario la modificación por el uso aun, pero en genero si es mas probable que se ocupe y necesitemos corregir.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelentes observaciones.</div>
# 
# [Volver a Contenidos](#back)

# ## Etapa 2. Preprocesamiento de datos <a id='data_preprocessing'></a>
# 
# El objetivo aquí es preparar los datos para que sean analizados.
# El primer paso es resolver cualquier problema con los encabezados. Luego podemos avanzar a los valores ausentes y duplicados. Empecemos.
# 
# Corrige el formato en los encabezados de la tabla.
# 

# ### Estilo del encabezado <a id='header_style'></a>
# Muestra los encabezados de la tabla (los nombres de las columnas):

# In[5]:


import pandas as pd # Importar pandas
df = pd.read_csv('/datasets/music_project_en.csv') # Leer el archivo y almacenarlo en df
print(df.columns)  # Muestra los nombres de las columnas
# aplicar .columns estamos tomando el dataframe y extraer los nombres de las columnas que el archivo presenta.


# Cambia los encabezados de la tabla de acuerdo con las reglas del buen estilo:
# * Todos los caracteres deben ser minúsculas.
# * Elimina los espacios.
# * Si el nombre tiene varias palabras, utiliza snake_case.

# Anteriormente, aprendiste acerca de la forma automática de cambiar el nombre de las columnas. Vamos a aplicarla ahora. Utiliza el bucle for para iterar sobre los nombres de las columnas y poner todos los caracteres en minúsculas. Cuando hayas terminado, vuelve a mostrar los encabezados de la tabla:

# In[6]:


for old_name in df.columns: # Este bucle toma las columnas del dataframe y transforma el nombre en minuscula tal como esta.
    name_lowered = old_name.lower() # se utiliza una nueva lista para mostrar los valores corregidos.
    print(name_lowered)     #muestra los valores como quedaron despues de la corrección y se puede ver que aun hay problemas.


# Ahora, utilizando el mismo método, elimina los espacios al principio y al final de los nombres de las columnas e imprime los nombres de las columnas nuevamente:

# In[7]:


for old_name in df.columns: # Bucle que corrige minusculas y espacios iniciales con finales.
    name_lowered = old_name.lower()  # variable que guarda las columnas en minuscula .
    name_stripped = name_lowered.strip() # variable que elimina los espacios al inicio y al final.
    print(name_stripped)  #muestra los nombres de las columnas corregidos. Aun se ve problemas a corregir.


# Necesitamos aplicar la regla de snake_case a la columna `userid`. Debe ser `user_id`. Cambia el nombre de esta columna y muestra los nombres de todas las columnas cuando hayas terminado.

# In[8]:


new_col_names = []  # creamos una lista donde estaran los nombres corregidos de todas las columnas 
for old_name in df.columns: # Bucle que corrige minusculas y espacios iniciales con finales.
        name_lowered = old_name.lower() # variable que guarda las columnas en minuscula .
        name_stripped = name_lowered.strip() # variable que elimina los espacios al inicio y al final.
        name_id= name_stripped.replace('userid', 'user_id').replace(' ', '_') # la variable que guarda el cambio con lenguaje snake_case a traves de replace
        new_col_names.append(name_id) #lista donde se agrega cada variable modificada 
print(new_col_names) # se muestra la lista con las variables ya corregidas.


# Comprueba el resultado. Muestra los encabezados una vez más:

# In[9]:


new_col_names = []
for old_name in df.columns: 
        name_lowered = old_name.lower()
        name_stripped = name_lowered.strip() 
        name_id= name_stripped.replace('userid', 'user_id').replace(' ', '_') 
        new_col_names.append(name_id)
df.columns = new_col_names  # se asiga la lista corregida a la variable que guarad las columnas del dataframe para poder tener el archivo con los datos corregidos.

print(df.columns)     # Comprobar el resultado: la lista de encabezados


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Esta muy bien. Como consejo, no arrastres el codigo cada vez que aplicas algo nuevo, o lo haces todo en una celda o en cada parte que hagas algo nuevo deja eso solo. Esto es debido a que el notebook ya carga la celda anterior. </div>

# [Volver a Contenidos](#back)

# ### Valores ausentes <a id='missing_values'></a>
#  Primero, encuentra el número de valores ausentes en la tabla. Debes utilizar dos métodos en una secuencia para obtener el número de valores ausentes.

# In[10]:


df.isna().sum()  # se aplican dos metodos de los cuales uno encuentra los valores ausentes y el otro los cuenta.


# No todos los valores ausentes afectan a la investigación. Por ejemplo, los valores ausentes en `track` y `artist` no son cruciales. Simplemente puedes reemplazarlos con valores predeterminados como el string `'unknown'` (desconocido).
# 
# Pero los valores ausentes en `'genre'` pueden afectar la comparación entre las preferencias musicales de Springfield y Shelbyville. En la vida real, sería útil saber las razones por las cuales hay datos ausentes e intentar recuperarlos. Pero no tenemos esa oportunidad en este proyecto. Así que tendrás que:
# * rellenar estos valores ausentes con un valor predeterminado;
# * evaluar cuánto podrían afectar los valores ausentes a tus cómputos;

# Reemplazar los valores ausentes en las columnas `'track'`, `'artist'` y `'genre'` con el string `'unknown'`. Como mostramos anteriormente en las lecciones, la mejor forma de hacerlo es crear una lista que almacene los nombres de las columnas donde se necesita el reemplazo. Luego, utiliza esta lista e itera sobre las columnas donde se necesita el reemplazo haciendo el propio reemplazo.

# In[11]:


replacement_columns = ['track','artist','genre'] #creamos una lista con las columnas que presentan ausentes y queremos modificar
for name in replacement_columns:  # Bucle recorre cada una de las columnas de la lista creada y si encuentra un valor ausente, va reemplazando los valores ausentes con 'unknown'
    df[name].fillna('unknown', inplace=True) # el metodo remplaza los valores ausentes por "unknown" y ocupamos inplace para no asignar una nueva variable.


# Ahora comprueba el resultado para asegurarte de que después del reemplazo no haya valores ausentes en el conjunto de datos. Para hacer esto, cuenta los valores ausentes nuevamente.

# In[12]:


df.isna().sum() # Contar valores ausentes
                #comprobamos que los cambios realizados fueron un exito al obtener valores cero en el conteo, osea no tenemos valores vacios.


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Buen bucle para remplazar los nulos y buenos comentarios en el codigo.</div>

# [Volver a Contenidos](#back)

# ### Duplicados <a id='duplicates'></a>
# Encuentra el número de duplicados explícitos en la tabla. Una vez más, debes aplicar dos métodos en una secuencia para obtener la cantidad de duplicados explícitos.

# In[13]:


print(df.duplicated().sum()) # este metodo nos encuentra los duplicados en las filas y ademas cuenta duplicados explícitos
print(df.head())  #este metodo nos muestra las primeras 5 filas del dataframe para ver 


# Ahora, elimina todos los duplicados. Para ello, llama al método que hace exactamente esto.

# In[14]:


df.drop_duplicates(inplace = True) # Eliminar duplicados explícitos 
df=df.reset_index()  #este metodo porne en orden los indices de la tabla


# Comprobemos ahora si eliminamos con éxito todos los duplicados. Cuenta los duplicados explícitos una vez más para asegurarte de haberlos eliminado todos:

# In[15]:


df.duplicated().sum() # Comprobar de nuevo si hay duplicados


# Ahora queremos deshacernos de los duplicados implícitos en la columna `genre`. Por ejemplo, el nombre de un género se puede escribir de varias formas. Dichos errores también pueden afectar al resultado.

# Para hacerlo, primero mostremos una lista de nombres de género únicos, ordenados en orden alfabético. Para ello:
# * Extrae la columna `genre` del DataFrame.
# * Llama al método que devolverá todos los valores únicos en la columna extraída.
# 

# In[16]:


print(sorted(df['genre'].unique())) # Entrega la lista de los nombres de géneros únicos que hay de forma ordenada
print(df['genre'].nunique())   # aqui entrega la cantidad de valores unicos que existen


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Excelente.</div>

# Busca en la lista para encontrar duplicados implícitos del género `hiphop`. Estos pueden ser nombres escritos incorrectamente o nombres alternativos para el mismo género.
# 
# Verás los siguientes duplicados implícitos:
# * `hip`
# * `hop`
# * `hip-hop`
# 
# Para deshacerte de ellos, crea una función llamada `replace_wrong_genres()` con dos parámetros:
# * `wrong_genres=`: esta es una lista que contiene todos los valores que necesitas reemplazar.
# * `correct_genre=`: este es un string que vas a utilizar como reemplazo.
# 
# Como resultado, la función debería corregir los nombres en la columna `'genre'` de la tabla `df`, es decir, remplazar cada valor de la lista `wrong_genres` por el valor en `correct_genre`.
# 
# Dentro del cuerpo de la función, utiliza un bucle `'for'` para iterar sobre la lista de géneros incorrectos, extrae la columna `'genre'` y aplica el método `replace` para hacer correcciones.

# In[17]:


def replace_wrong_genres(d,column ,wrong_genres,correct_genre): # Función para reemplazar duplicados implícitos
    for wrong_genre in wrong_genres :
        d[column].replace(wrong_genre,correct_genre, inplace = True)
    return d


# Ahora, llama a `replace_wrong_genres()` y pásale tales argumentos para que retire los duplicados implícitos (`hip`, `hop` y `hip-hop`) y los reemplace por `hiphop`:

# In[18]:


wrong_genres = ['hip','hop','hip-hop']     # Eliminar duplicados implícitos
correct_genre = 'hiphop'
df= replace_wrong_genres(df,'genre',wrong_genres,correct_genre)  # df queda con los valores corregidos 



# Asegúrate de que los nombres duplicados han sido eliminados. Muestra la lista de valores únicos de la columna `'genre'` una vez más:

# In[19]:


print(sorted(df['genre'].unique()))    # Comprobación de duplicados implícitos entregando el listado
print(df['genre'].nunique())  # comprobación de la cantidad de duplicados implicitos


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy bien utilizando el nunique().</div>

# ### Tus observaciones <a id='data_preprocessing_conclusions'></a>
# 
# `Describe brevemente lo que has notado al analizar duplicados, cómo abordaste sus eliminaciones y qué resultados obtuviste.`

# Se pudo encontrar que habian 3826 valores duplicados , de los cuales se identiifcaron y fueron eliminados en la tabla para dejar todo limpio, pero ademas se pudo identificar que habia una cantidad de valores unicos correspondiente a 269 valores que al observar la data se vio que en el genero hip-hop estaban mal escritos hip , hop y hip-hop por lo que al coregir los nombres de los valores unicos entrego que en realidad hay 266 valores unicos en el archivo. Al eliminar los valores duplicados , se tuvo que corregir los indices de la tabla para que quedaran correlativos y obtener una dataframe sin valores duplicados y la cantidad correcta de valores unicos.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Muy buenas observaciones.</div>

# [Volver a Contenidos](#back)

# ## Etapa 3. Prueba de hipótesis <a id='hypothesis'></a>

# ### Hipótesis: comparar el comportamiento del usuario o la usuaria en las dos ciudades <a id='activity'></a>

# La hipótesis afirma que existen diferencias en la forma en que los usuarios y las usuarias de Springfield y Shelbyville consumen música. Para comprobar esto, usa los datos de tres días de la semana: lunes, miércoles y viernes.
# 
# * Agrupa a los usuarios y las usuarias por ciudad.
# * Compara el número de canciones que cada grupo reprodujo el lunes, el miércoles y el viernes.
# 

# Realiza cada cálculo por separado.
# 
# El primer paso es evaluar la actividad del usuario en cada ciudad. Recuerda las etapas dividir-aplicar-combinar de las que hablamos anteriormente en la lección. Tu objetivo ahora es agrupar los datos por ciudad, aplicar el método apropiado para contar durante la etapa de aplicación y luego encontrar la cantidad de canciones reproducidas en cada grupo especificando la columna para obtener el recuento.
# 
# A continuación se muestra un ejemplo de cómo debería verse el resultado final:
# `df.groupby(by='....')['column'].method()`Realiza cada cálculo por separado.
# 
# Para evaluar la actividad de los usuarios y las usuarias en cada ciudad, agrupa los datos por ciudad y encuentra la cantidad de canciones reproducidas en cada grupo.
# 
# 

# In[20]:


time_city = df.groupby('city')['time'].count()    # Contar las canciones reproducidas en cada ciudad
print(time_city) # esta variable muestra la cantidad de canciones que se escuchan por ciudad los ususarios.


# `Comenta tus observaciones aquí`

# Los usuarios de la ciudad springfield escuchan una cantidad mayor de canciones que la ciudad shelbyville, considerando el tiempo  total de canciones vemos que el 69,77% de ususarios de sprinfield son los que escuchan mas tiempo canciones.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Perfecto.</div>

# Ahora agrupemos los datos por día de la semana y encontremos el número de canciones reproducidas el lunes, miércoles y viernes. Utiliza el mismo método que antes, pero ahora necesitamos una agrupación diferente.
# 

# In[21]:


time_of_day= df.groupby('day')['time'].count() # Calcular las canciones reproducidas en cada uno de los tres días
print(time_of_day)


# `Comenta tus observaciones aquí`

# Vemos que los dias viernes , es el dia que mayor tiempo de canciones escuchan los usuarios, seguidos por el dia lunes y luego miercoles.

# Ya sabes cómo contar entradas agrupándolas por ciudad o día. Ahora necesitas escribir una función que pueda contar entradas según ambos criterios simultáneamente.
# 
# Crea la función `number_tracks()` para calcular el número de canciones reproducidas en un determinado día **y** ciudad. La función debe aceptar dos parámetros:
# 
# - `day`: un día de la semana para filtrar. Por ejemplo, `'Monday'` (lunes).
# - `city`: una ciudad para filtrar. Por ejemplo, `'Springfield'`.
# 
# Dentro de la función, aplicarás un filtrado consecutivo con indexación lógica.
# 
# Primero filtra los datos por día y luego filtra la tabla resultante por ciudad.
# 
# Después de filtrar los datos por dos criterios, cuenta el número de valores de la columna 'user_id' en la tabla resultante. Este recuento representa el número de entradas que estás buscando. Guarda el resultado en una nueva variable y devuélvelo desde la función.

# In[22]:


def number_track(day,city)  :                 # Declara la función number_tracks() con dos parámetros: day= y city=.      
    filtered_by_day = df[df['day'] == day]    # Almacena las filas del DataFrame donde el valor en la columna 'day' es igual al parámetro day=
    filtered_by_day_and_city = filtered_by_day[filtered_by_day['city'] == city]   # Filtra las filas donde el valor en la columna 'city' es igual al parámetro city=
    tracks_count= filtered_by_day_and_city['user_id'].count() # Extrae la columna 'user_id' de la tabla filtrada y aplica el método count()
    return tracks_count   # Devolve el número de valores de la columna 'user_id'
   

    


# Llama a `number_tracks()` seis veces, cambiando los valores de los parámetros para que recuperes los datos de ambas ciudades para cada uno de los tres días.

# In[23]:


Shelbyville_monday= number_track('Monday','Shelbyville')# El número de canciones reproducidas en Shelbyville el lunes
print(Shelbyville_monday)


# In[24]:


Springfield_monday= number_track('Monday','Springfield')# El número de canciones reproducidas en Shelbyville el lunes
print(Springfield_monday)


# In[25]:


Springfield_wednesday=number_track('Wednesday','Springfield')# El número de canciones reproducidas en Springfield el miércoles
print(Springfield_wednesday)


# In[26]:


Shelbyville_wednesday=number_track('Wednesday','Shelbyville')# El número de canciones reproducidas en Shelbyville el miércoles
print(Shelbyville_wednesday)


# In[27]:


Springfield_friday=number_track('Friday','Springfield')# El número de canciones reproducidas en Springfield el viernes
print(Springfield_friday)


# In[28]:


Shelbyville_friday=number_track('Friday','Shelbyville')# El número de canciones reproducidas en Shelbyville el viernes
print(Shelbyville_friday)


# **Conclusiones**
# 
# `Comenta si la hipótesis es correcta o se debe rechazar. Explica tu razonamiento.`

# La hipotesis es correcta, hay diferenciación entre la actividad de los ususarios de acuerdo al dia de la semana y la ciudad de cuales son.

# [Volver a Contenidos](#back)

# # Conclusiones <a id='end'></a>

# `Resume aquí tus conclusiones sobre la hipótesis.`

# Podemos concluir de las actividades realizadas por los ususarios ,presentan diferencias al momento de esuchcar las canciones, observandose que los usuarios de la ciudad de springfield escuchan un mayor tiempo las canciones para los tres dias , lunes , miercoles y viernes , en comparación a shelbyville que escucha menos canciones que los ususarios de la otra ciudad y estos son menores a la vez por cada dia evaluado.
# Vemos que del total de canciones de todos los usuarios de ambas ciudades tenemos un total de 61253, que el 69,77% corresponde a la ciudad de springfield y que el 37,3% de las canciones escuchadas en esa ciudad lo realizan el dia viernes.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
# 
# Las conclusiones estan muy bien, explicas claramente el resultado.</div>

# ### Nota
# En proyectos de investigación reales, la prueba de hipótesis estadística es más precisa y cuantitativa. También ten en cuenta que no siempre se pueden sacar conclusiones sobre una ciudad entera a partir de datos de una sola fuente.
# 
# Aprenderás más sobre la prueba de hipótesis en el sprint de análisis estadístico de datos.

# [Volver a Contenidos](#back)
