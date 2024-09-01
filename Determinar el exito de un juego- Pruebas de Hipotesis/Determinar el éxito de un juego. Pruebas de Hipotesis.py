

# Paso 1. Abre el archivo de datos y estudia la información general


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import scipy.stats as stats

# Cargar el dataset
games = pd.read_csv('C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS BOOTCAMP/Determinar el exito de un juego- Pruebas de Hipotesis/games.csv')

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(games.head())

# Información general del dataset
print("\nInformación general del dataset:")
print(games.info())

print(games.isnull().sum())


# Hay dos nulos en Name ( quizas hay que eliminarlos) , year_of_release es flot (quizas convertir en tipo fecha) , genre tiene dos nulos 
# ( quizas son los mismos que name, posible elimnacion). Critic_score, user_core, rating son los que mayor nulos presentan, hay que trabajarlos. 
# Cambiar mayusculas a minusculas
# user_score y raiting nose si es bueno aun cambiarlos de tipo a float aun.

# Paso 2. Preparar los datos

# Reemplazar los nombres de las columnas por minúsculas
games.columns = games.columns.str.lower()

# Convertir los datos en las columnas de tipo objeto a minúsculas
for col in games.select_dtypes(include=['object']).columns:
    games[col] = games[col].str.lower()

print("\nNombres de las columnas después de la modificación:")
print(games.columns)


# Verificar valores duplicados
duplicated_rows = games[games.duplicated()]
print(f"Filas duplicadas encontradas: {len(duplicated_rows)}")

# Convertir la columna 'years_of_release' en formato datetime, para que este la información mas acorde y los graficos sean mejor.
# Voy a crear otra columna con la modificación
games['years_date']=pd.to_datetime(games['year_of_release'], format ='%Y')

# Ordenar y contar los valores de la columna user_score considerando los Nan de acuerdo a info()
games['user_score'].value_counts(dropna=False).sort_index()

# Contar los valores de rating
games['rating'].value_counts(dropna=False)

# Reemplazar 'tbd' por nan en user_score. Si pongo otro valor puedo sesgar los datos
games['user_score_fixed'] = np.where(games['user_score'] == 'tbd', 'nan', games['user_score'])

# Convertir user_score a tipo float
games['user_score_fixed'] = games['user_score_fixed'].astype(float)

# Descripción estadística de las columnas critic_score y user_score
print("\nDescripción estadística de critic_score y user_score:")
print(games[['critic_score', 'user_score_fixed']].describe())

# Filtrar las filas con NaN en 'genre' y 'name'
nan_rows = games[games['genre'].isna() | games['name'].isna()]

# Mostrar las filas con NaN en 'genre' y 'name'
print("Filas con NaN en 'genre' o 'name':")
print(nan_rows)
# Eliminar las filas con NaN en 'genre' y 'name'  son solo dos filas de los cuales , la mayoria de los datos son nan
# Por lo que no entregaria un mayor cambio en la data de analisis 

games.dropna(subset=['genre', 'name'], inplace=True)

# Los valores nan que se encuentran en Rating , dado a que corresponden a una clasificación , estimo mejor ponerlos en 'RP' 
# que significa aun sin clasificar.

# Crear una nueva columna llamada 'rating_modified' con los valores de 'rating' pero rellenando los NaN con 'RP'
games['rating_fixed'] = games['rating'].fillna('rp')
# Contar los valores de rating
games['rating_fixed'].value_counts(dropna=False)

# Calcular las ventas totales
games['total_sales'] = games['na_sales'] + games['eu_sales'] + games['jp_sales'] + games['other_sales']

# Número de géneros
num_genres = games['genre'].nunique()
print(f"Número de géneros: {num_genres}")

# Número de plataformas en cada género
platforms_per_genre = games.groupby('genre')['platform'].nunique().reset_index()
platforms_per_genre.columns = ['genre', 'num_platforms']
print("Número de plataformas por género:")
print(platforms_per_genre)
print()
# Calcular el número de plataformas únicas
num_platforms = games['platform'].nunique()
print(f"Número de plataformas: {num_platforms}")
print()
# Obtener el listado de plataformas únicas
unique_platforms = games['platform'].unique()
print("Listado de plataformas únicas:")
print(unique_platforms)

# Extraer el año de 'years_date'
games['year'] = games['years_date'].dt.year

# Calcular medianas por grupo (género y plataforma)
critic_median = games.groupby(['genre', 'platform'])['critic_score'].median().reset_index()
user_median = games.groupby(['genre', 'platform'])['user_score_fixed'].median().reset_index()

print(critic_median)
print(user_median)

# Renombrar las columnas para evitar conflictos
critic_median.rename(columns={'critic_score': 'critic_score_median'}, inplace=True)
user_median.rename(columns={'user_score_fixed': 'user_score_median'}, inplace=True)

# Unir las medianas al dataset original
games = games.merge(critic_median, on=['genre', 'platform'], how='left')
games = games.merge(user_median, on=['genre', 'platform'], how='left')

# Imputar los valores NaN en critic_score y user_score_fixed
games['critic_score_fixed'] = games['critic_score'].fillna(games['critic_score_median'])
games['user_score_fixed'] = games['user_score_fixed'].fillna(games['user_score_median'])

# Filtrar los datos con valores NaN en 'critic_score_fixed' y 'user_score_fixed'
nan_critic_score_data = games[games['critic_score_fixed'].isna()]
nan_user_score_data = games[games['user_score_fixed'].isna()]

# Verificar si tienen ventas
nan_critic_score_data_with_sales = nan_critic_score_data[nan_critic_score_data['total_sales'] > 0]
nan_user_score_data_with_sales = nan_user_score_data[nan_user_score_data['total_sales'] > 0]

# Agrupar por género y plataforma para ver la distribución de ventas en critic_score_fixed
grouped_sales_critic = nan_critic_score_data_with_sales.groupby(['genre', 'platform']).agg({'total_sales': ['count', 'sum'], 'year': 'median'}).reset_index()
grouped_sales_critic.columns = ['genre', 'platform', 'count_sales', 'sum_sales', 'median_year']

# Agrupar por género y plataforma para ver la distribución de ventas en user_score_fixed
grouped_sales_user = nan_user_score_data_with_sales.groupby(['genre', 'platform']).agg({'total_sales': ['count', 'sum'], 'year': 'median'}).reset_index()
grouped_sales_user.columns = ['genre', 'platform', 'count_sales', 'sum_sales', 'median_year']

# Mostrar los datos agrupados para critic_score_fixed
print("Datos agrupados para critic_score_fixed:")
print(grouped_sales_critic)

# Mostrar los datos agrupados para user_score_fixed
print("Datos agrupados para user_score_fixed:")
print(grouped_sales_user)

# Contar el número total de filas en el dataset original
total_datos = games.shape[0]

# Contar el número de filas con 'year_of_release' como NaN
total_nan_years = games['year_of_release'].isna().sum()

# Calcular el porcentaje de datos eliminados
porcentaje_datos_eliminados = (total_nan_years / total_datos) * 100

# Mostrar los resultados
print(f"Número total de datos eliminados: {total_nan_years}")
print(f"Porcentaje de datos eliminados: {porcentaje_datos_eliminados:.2f}%")


# Los valores nan que se encuentran presenten en fecha son muy pocos de acuerdo al porcentaje encontrado , de acuerdo a esto se decidio eliminarlos debido 
# a que muchos calculos se considera el año para evaluar.

# Calcular medianas globales
global_median_critic_score = games['critic_score_fixed'].median()
global_median_user_score = games['user_score_fixed'].median()

# Rellenar los NaN restantes con las medianas globales
games['critic_score_fixed'].fillna(global_median_critic_score, inplace=True)
games['user_score_fixed'].fillna(global_median_user_score, inplace=True)

# Verificar si quedan valores NaN en las nuevas columnas
nan_critic_score_fixed_final = games['critic_score_fixed'].isna().sum()
nan_user_score_fixed_final = games['user_score_fixed'].isna().sum()

# Mostrar los resultados finales
nan_critic_score_fixed_final, nan_user_score_fixed_final


# Se cambio todo a minusculas
# Se cambiaron de tipo de datos, años por datetime, user_score a float.
# Se vieron si habian duplicados para eliminarlos ( no habian)
# Se crearon las columnas raiting_fixed, year_date, critic_score_fixed y user_score_fixed. Fueron creadas con las modificaciones correspondientes que se le hizo a cada columna para no modificar la columna original. 
# De acuerdo a los valores ausentes, los nan de raiting fueron cambiados por rp , lo que corresponde a la clasificación de no determinados ( osea falta que los clasifiquen). Para user_score , presenta valores nan y tbd, dado a que tbd corresponde a una evaluación de los usuarios que aun no esta determinada no podemos agregarselos a nan por lo que se decidio ponerles -1 hasta que presentaran clasificación.
# Se eliminaron 2 valores que no eran muy representativos correspondientes a dos filas que presentaban nan en genero y name.
# Para los valores nan en crisitc_score y user_score , se evaluaron las medianas de cada columna de acuerdo al genero y plataforma y se remplazaron los nan por esas medianas. En critic_score_fixed y user_score_fixed aun quedan nan, de acuerdo a esto se remplazan los nan por la mediana global
# Pueden haber otras modificaciones en valores o eliminación que se evaluaran mas adelante.

# Paso 3: Analizar los datos

# Mira cuantos juegos fueron lanzados en diferentes años ¿Son significativos los datos observados de cada periodo?
# 1. Análisis de juegos lanzados en diferentes años
games_per_year = games.groupby('years_date')['name'].count()
plt.figure(figsize=(10, 6))
plt.plot(games_per_year.index, games_per_year.values, marker='o')
plt.title('Número de juegos lanzados por año')
plt.xlabel('Año')
plt.ylabel('Número de juegos')
plt.grid(True)
plt.show()


# Crecimiento Inicial:
# 
# Desde 1980 hasta mediados de la década de 1990, se observa un crecimiento lento pero constante en el número de juegos lanzados.
# Incremento Significativo a Finales de los 90 y Principios de los 2000:
# 
# A partir de mediados de la década de 1990, hay un aumento más pronunciado en el número de juegos lanzados. Este periodo coincide con la popularización de consolas como PlayStation y Nintendo 64.
# Pico Máximo en 2008-2009:
# 
# El gráfico muestra un pico máximo en el número de juegos lanzados en 2008-2009. Este periodo coincide con la popularidad de consolas como PlayStation 3, Xbox 360 y Nintendo Wii, así como un auge en el desarrollo de juegos para dispositivos móviles y PC.
# 
# Declive Posterior:
# Después del pico de 2008-2009, se observa un declive en el número de juegos lanzados, aunque sigue siendo relativamente alto en comparación con las décadas anteriores. Esto puede deberse a varios factores, incluyendo cambios en la industria, la transición a nuevas plataformas, y la consolidación de estudios de desarrollo.
# 
# Estabilización Reciente:
# A partir de 2012, el número de juegos lanzados parece estabilizarse con algunas fluctuaciones menores.
# 
# ¿Son significativos los datos observados de cada periodo?
# 
# Sí, los datos observados de cada periodo son significativos por varias razones:Tendencias de Mercado,Impacto de Nuevas Tecnologías, Ciclos Económicos, Planificación Estratégica.


# Observa como varian las ventas de una plataforma a otra.
# Calcular las ventas totales de una plataforma.
ventas_por_plataforma = games.groupby('platform')['total_sales'].sum().sort_values(ascending=False)

# Graficar las ventas totales por plataforma
plt.figure(figsize=(12, 6))
ventas_por_plataforma.plot(kind='bar', color='skyblue')
plt.title('Ventas totales por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Ventas totales (en millones)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# Se puede decir que las plataformas con las mayores ventas, y por lo tanto las "top de ventas", son las primeras seis PS2 (PlayStation 2), X360 (Xbox 360), PS3 (PlayStation 3), Wii, DS (Nintendo DS), PS (PlayStation original)
# Estas plataformas destacan claramente en el gráfico debido a sus altas ventas totales en comparación con las demás. La PS2 es la líder indiscutible, seguida por la Xbox 360 y la PS3. La Wii, DS y la PlayStation original también muestran ventas sustanciales, lo que las coloca en el grupo de plataformas con más éxito en términos de ventas.

# Filtrar los datos de Nintendo DS
ds_sales = games[games['platform'] == 'ds']

# Agrupar las ventas por año
ds_sales_by_year = ds_sales.groupby(ds_sales['years_date'].dt.year)['total_sales'].sum().reset_index()

# Crear una gráfica de las ventas de Nintendo DS por año
plt.figure(figsize=(10, 6))
plt.plot(ds_sales_by_year['years_date'], ds_sales_by_year['total_sales'], marker='o')
plt.title('Ventas Totales de Nintendo DS por Año')
plt.xlabel('Año')
plt.ylabel('Ventas Totales (en millones)')
plt.grid(True)
plt.show()

# Mostrar los datos
print(ds_sales_by_year)

# Observando la gráfica, parece que el punto de 1985 representa una venta muy baja en comparación con el resto de los datos. Esto sugiere que podría tratarse de un error o de un valor atípico que no refleja correctamente el rendimiento de las ventas de Nintendo DS.


# Filtrar los datos de Nintendo DS y eliminar el año 1985
games = games[~((games['platform'] == 'ds') & (games['year_of_release'] == 1985))]
ds_sales = games[games['platform'] == 'ds']

# Agrupar las ventas por año
ds_sales_by_year = ds_sales.groupby(ds_sales['years_date'].dt.year)['total_sales'].sum().reset_index()

# Crear una gráfica de las ventas de Nintendo DS por año
plt.figure(figsize=(10, 6))
plt.plot(ds_sales_by_year['years_date'], ds_sales_by_year['total_sales'], marker='o')
plt.title('Ventas Totales de Nintendo DS por Año')
plt.xlabel('Año')
plt.ylabel('Ventas Totales (en millones)')
plt.grid(True)
plt.show()

# Mostrar los datos
print(ds_sales_by_year)

# Elige las plataformas con mayores ventas totales y construye una distribución basada en los datos de cada año.

# Selección de las plataformas con mayores ventas totales
top_platforms = games.groupby('platform')['total_sales'].sum().sort_values(ascending=False).head(6)

# Construcción de la distribución de ventas por año para las plataformas seleccionadas
plt.figure(figsize=(12, 6))
for platform in top_platforms.index:
    platform_sales = games[games['platform'] == platform].groupby('years_date')['total_sales'].sum()
    platform_sales.plot(label=platform)
plt.title('Ventas totales por año para las 6 principales plataformas')
plt.xlabel('Año')
plt.ylabel('Ventas totales')
plt.legend()
plt.show() 

# Crear una columna con solo el año a partir de 'years_date'
games['year'] = games['years_date'].dt.year 

# Calcular el número de juegos lanzados por plataforma y año
juegos_por_año_plataforma = games.pivot_table(index='year', columns='platform', values='name', aggfunc='count').fillna(0)

# Graficar el número de juegos lanzados por plataforma y año (barras agrupadas)
juegos_por_año_plataforma.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')
plt.title('Número de juegos lanzados por año y plataforma')
plt.xlabel('Año')
plt.ylabel('Número de juegos')
plt.legend(title='Plataforma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.show()

# Calcular las ventas totales por plataforma y año
ventas_por_plataforma_año = games.pivot_table(index='year', columns='platform', values='total_sales', aggfunc='sum').fillna(0)

# Graficar las ventas totales por plataforma y año (barras apiladas)
ventas_por_plataforma_año.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')
plt.title('Ventas totales por plataforma y año')
plt.xlabel('Año')
plt.ylabel('Ventas totales (en millones)')
plt.legend(title='Plataforma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.show()


# Dado que el enunciado del proyecto menciona datos hasta 2016 y asume una planificación para 2017, elegir 2015 permite analizar las plataformas con datos recientes sin estar demasiado cerca del límite de los datos disponibles.Ademas el ennunciado menciona que los datos del 2016 no estan completos por lo que 2015 es una buena fecha actual.

# Calcular el año de primera y última venta por plataforma
año_primera_venta = games.groupby('platform')['years_date'].min().reset_index()
año_ultima_venta = games.groupby('platform')['years_date'].max().reset_index()

# Unir los datos de primera y última venta
tiempo_vida_plataformas = pd.merge(año_primera_venta, año_ultima_venta, on='platform')
tiempo_vida_plataformas.columns = ['platform', 'año_primera_venta', 'año_ultima_venta']

# Calcular la duración de vida de cada plataforma
tiempo_vida_plataformas['duracion'] = tiempo_vida_plataformas['año_ultima_venta'].dt.year - tiempo_vida_plataformas['año_primera_venta'].dt.year

print("Duración de vida de cada plataforma (en años):")
print(tiempo_vida_plataformas)

# Calcular la media de duración de vida de las plataformas
media_duracion = tiempo_vida_plataformas['duracion'].mean()
print(f"Duración media de vida de las plataformas: {media_duracion:.2f} años")


# Duración de Vida de las Plataformas: 7 años
# El análisis del tiempo de aparición y desaparición de las plataformas nos muestra cuánto tiempo suelen estar activas en el mercado antes de volverse obsoletas. La media de la duración de vida de las plataformas proporciona una idea general del ciclo de vida típico de las consolas de videojuegos.
# Determina para qué período debes tomar datos. Para hacerlo mira tus respuestas a las preguntas anteriores. Los datos deberían permitirte construir un modelo para 2017.

# Para capturar las tendencias recientes y relevantes para construir un modelo para 2017, utilizaremos 
# datos de los últimos 8 años, es decir, de 2010 a 2016.

# Filtrar los datos del período 2010-2016
games_relevantes = games[(games['years_date'].dt.year >= 2010) & (games['years_date'].dt.year <= 2016)]

# Mostrar los primeros registros de los datos filtrados
print(games_relevantes.head())
print()
# Verificar el rango de años en los datos filtrados
print(games_relevantes['years_date'].dt.year.min(), games_relevantes['years_date'].dt.year.max())


# Período Seleccionado: 2010-2016
# Este período abarca las tendencias más recientes en el mercado de videojuegos.
# Se excluyen los datos de plataformas que ya no son relevantes.
# Duración Adecuada: 7 años proporcionan un período adecuado para observar la evolución de las ventas y la popularidad de las plataformas sin ser demasiado amplio.
# Este período de datos es relevante y adecuado para construir un modelo predictivo para el año 2017, capturando las tendencias y comportamientos recientes del mercado de videojuegos.

# Calcular las ventas totales por plataforma en el período 2010-2016
ventas_por_plataforma = games_relevantes.groupby('platform')['total_sales'].sum().sort_values(ascending=False)

# Mostrar las plataformas líderes en ventas
print("Plataformas líderes en ventas (2010-2016):")
print(ventas_por_plataforma)

# Analizar las ventas anuales por plataforma para identificar tendencias de crecimiento y declive
plt.figure(figsize=(12, 6))
for platform in ventas_por_plataforma.index:
    platform_sales = games_relevantes[games_relevantes['platform'] == platform].groupby(games_relevantes['years_date'].dt.year)['total_sales'].sum()
    plt.plot(platform_sales.index, platform_sales.values, marker='o', label=platform)
plt.title('Ventas anuales por plataforma (2010-2016)')
plt.xlabel('Año')
plt.ylabel('Ventas totales (en millones)')
plt.legend()
plt.grid(True)
plt.show()


# 1. ¿Qué plataformas son líderes en ventas?
# Las plataformas líderes en ventas totales durante el período 2010-2016 son:
# 
# PS3 (azul): La línea muestra altas ventas especialmente en 2010 y 2011, aunque decrece después de estos años.
# X360 (naranja): Similar a PS3, tiene ventas altas en 2010 y 2011, pero decrece posteriormente.
# PS4 (verde): Muestra un aumento significativo en ventas desde su lanzamiento en 2013 hasta 2016.
# 3DS (rojo): Tiene ventas relativamente altas en el período 2010-2016.
# Wii (morado): Aunque comienza con altas ventas en 2010, decrece rápidamente en los años siguientes.
# Estas plataformas han registrado las ventas totales más altas en el período mencionado.
# 
# 2. ¿Cuáles crecen y cuáles se reducen?
# Analizando el gráfico de ventas anuales por plataforma:
# 
# Plataformas en Crecimiento:
# PS4: Muestra un crecimiento significativo desde su lanzamiento en 2013 hasta 2016.
# Xone: Aunque las ventas no son tan altas como PS4, se ve una tendencia al alza desde 2013 hasta 2016.
# 
# Plataformas en Declive:
# PS3: Después de un pico en 2010, muestra una tendencia decreciente.
# X360: Similar a PS3, con un declive gradual después de 2011.
# Wii: Fuerte declive después de 2010.
# DS: Mostrando una disminución continua después de 2010.
# PSP: También en declive hacia el final del período.
# 
# 3. Elige varias plataformas potencialmente rentables.
# PS4: Es claramente la plataforma más prometedora con un crecimiento significativo. Invertir en esta plataforma parece una apuesta segura.
# Xone: Aunque su crecimiento no es tan fuerte como el de PS4, sigue siendo una plataforma en ascenso y podría ser rentable.
# 3DS: Aunque muestra fluctuaciones, aún tiene ventas significativas y podría ser considerada para estrategias específicas.
# 
# Conclusión
# Las plataformas PS4 y Xone deben ser el principal enfoque debido a su crecimiento y potencial de mercado. Monitorear 3DS puede ofrecer oportunidades adicionales, mientras que el enfoque en plataformas en declive debe ser minimizado para optimizar recursos y maximizar beneficios.

# Crear un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma
plt.figure(figsize=(12, 8))
sns.boxplot(x='platform', y='total_sales', data=games_relevantes)
plt.title('Diagrama de caja de las ventas globales por plataforma (2010-2016)')
plt.xlabel('Plataforma')
plt.ylabel('Ventas globales (en millones)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Calcular el rango intercuartílico (IQR) para cada plataforma
Q1 = games_relevantes.groupby('platform')['total_sales'].quantile(0.25)
Q3 = games_relevantes.groupby('platform')['total_sales'].quantile(0.75)
IQR = Q3 - Q1

# Crear un DataFrame con Q1, Q3 e IQR para cada plataforma
iqr_data = pd.DataFrame({'Q1': Q1, 'Q3': Q3, 'IQR': IQR}).reset_index()

# Fusionar el DataFrame iqr_data con games_relevantes
games_relevantes = games_relevantes.merge(iqr_data, on='platform')

# Identificar outliers
outliers = games_relevantes[
    (games_relevantes['total_sales'] < (games_relevantes['Q1'] - 1.5 * games_relevantes['IQR'])) |
    (games_relevantes['total_sales'] > (games_relevantes['Q3'] + 1.5 * games_relevantes['IQR']))
]

print("Outliers identificados:")
print(outliers)

# Opcional: eliminar outliers del dataset
games_sin_outliers = games_relevantes.drop(outliers.index)

# Crear un DataFrame sin los outliers
games_sin_outliers = games_relevantes.drop(outliers.index)

# Verificar el tamaño del nuevo DataFrame
print(f"Tamaño del DataFrame original: {games_relevantes.shape}")
print(f"Tamaño del DataFrame sin outliers: {games_sin_outliers.shape}")

# Rehacer el diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma
plt.figure(figsize=(12, 8))
sns.boxplot(x='platform', y='total_sales', data=games_sin_outliers)
plt.title('Diagrama de caja de las ventas globales por plataforma (2010-2016) - Sin Outliers')
plt.xlabel('Plataforma')
plt.ylabel('Ventas globales (en millones)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# - ¿Son Significativas las Diferencias en las Ventas?
# Sí, las diferencias en las ventas entre plataformas son significativas. El gráfico de caja muestra que algunas plataformas tienen una mediana de ventas mucho más alta que otras, y también hay diferencias en la amplitud de los rangos intercuartiles (IQR). Aquí algunos puntos clave:
# 
# PS3 y X360: Tienen las mayores medianas de ventas, lo que indica que, en promedio, los juegos en estas plataformas venden más que en otras. Además, muestran una amplia distribución, lo que sugiere una variabilidad considerable en las ventas de los juegos.
# PS4 y Xone: También presentan medianas relativamente altas, aunque más bajas que las de PS3 y X360. Sin embargo, muestran una tendencia de aumento en las ventas, como se vio en el análisis previo.
# Wii y 3DS: Tienen medianas y rangos de ventas más bajos en comparación con las plataformas mencionadas anteriormente.
# PC, PSP, PSV y PS2: Estas plataformas tienen las medianas de ventas más bajas, indicando ventas significativamente menores en comparación con las plataformas líderes.
# - ¿Qué sucede con las ventas promedio en varias plataformas?
# El diagrama de caja permite observar la mediana de ventas (la línea dentro de cada caja) y la dispersión de los datos. Los puntos clave son:
# 
# PS3 y X360: Muestran las ventas promedio (medianas) más altas entre todas las plataformas. Las cajas más grandes y los bigotes más largos indican que, aunque tienen juegos con altas ventas, también hay una variabilidad considerable.
# PS4 y Xone: Tienen ventas promedio relativamente altas, pero con una distribución más centrada y menos dispersa que PS3 y X360.
# Wii y 3DS: Sus ventas promedio son más bajas, con cajas más pequeñas que indican una menor variabilidad en las ventas.
# PC, PSP, PSV y PS2: Las ventas promedio son las más bajas, reflejando una menor popularidad o base de usuarios en comparación con otras plataformas.
# - Descripción breve de los hallazgos:
# 
# Significativas diferencias de ventas: Las diferencias en las ventas entre plataformas son significativas, con PS3 y X360 liderando en términos de ventas medianas.
# Ventas promedio altas en plataformas populares: PS3 y X360 tienen las ventas promedio más altas, seguidas por PS4 y Xone, lo que refleja su popularidad y éxito en el mercado durante el período 2010-2016.
# Variabilidad en las ventas: Hay una alta variabilidad en las ventas dentro de las plataformas líderes, lo que sugiere que aunque algunos juegos tienen un éxito tremendo, otros no lo hacen tan bien.
# Plataformas menos rentables: PC, PSP, PSV y PS2 tienen las ventas promedio más bajas, indicando que podrían no ser tan rentables como las plataformas líderes.

# Mira cómo las reseñas de usuarios y profesionales afectan las ventas de una plataforma popular (tu elección). 
# Crea un gráfico de dispersión y calcula la correlación entre las reseñas y las ventas. Saca conclusiones.


# Vamos a analizar cómo las reseñas de usuarios y profesionales afectan las ventas en una plataforma popular. 
# Elegiré la plataforma PS4 para este análisis debido a su popularidad y ventas significativas.

# Calcular las ventas totales de PS4 en el período 2010-2016
ps4_total_sales = games_relevantes[games_relevantes['platform'] == 'ps4']['total_sales'].sum()
print(f"Ventas totales de PS4 (2010-2016): {ps4_total_sales:.2f} millones")

# Comparar las ventas totales de PS4 con otras plataformas líderes
ventas_por_plataforma = games_relevantes.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
print(ventas_por_plataforma)

# Filtrar los datos para la plataforma PS4
ps4_games = games_relevantes[games_relevantes['platform'] == 'ps4']

# Mostrar los primeros registros de los datos filtrados
print(ps4_games.head())


# Gráfico de dispersión para las reseñas de los críticos
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=ps4_games, x='critic_score_fixed', y='total_sales', alpha=0.5)
plt.title('Critic Score vs Total Sales (PS4)')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales (millions)')
plt.grid(True)

# Gráfico de dispersión para las reseñas de los usuarios
plt.subplot(1, 2, 2)
sns.scatterplot(data=ps4_games, x='user_score_fixed', y='total_sales', alpha=0.5)
plt.title('User Score vs Total Sales (PS4)')
plt.xlabel('User Score')
plt.ylabel('Total Sales (millions)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calcular la correlación
correlation_critic = ps4_games['critic_score_fixed'].corr(ps4_games['total_sales'])
correlation_user = ps4_games['user_score_fixed'].corr(ps4_games['total_sales'])

print(f"Correlación entre Critic Score y Total Sales (PS4): {correlation_critic:.2f}")
print(f"Correlación entre User Score y Total Sales (PS4): {correlation_user:.2f}")


# Confirmar el número de registros en ps4_games
print(f"Número de registros en PS4 Games: {len(ps4_games)}")

# Confirmar que no se han eliminado registros para el análisis de correlación
print(f"Número total de registros en games_relevantes: {len(games_relevantes)}")


# Definir el rango sin outliers
Q1 = games_relevantes['total_sales'].quantile(0.25)
Q3 = games_relevantes['total_sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrar los datos para eliminar los outliers
games_relevantes_no_outliers = games_relevantes[
    (games_relevantes['total_sales'] >= lower_bound) &
    (games_relevantes['total_sales'] <= upper_bound)
]

# Filtrar los datos de PS4 sin outliers
ps4_games_no_outliers = games_relevantes_no_outliers[games_relevantes_no_outliers['platform'] == 'ps4']

# Gráfico de dispersión sin outliers
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=ps4_games_no_outliers, x='critic_score_fixed', y='total_sales', alpha=0.5)
plt.title('Critic Score vs Total Sales (PS4) - Sin Outliers')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales (millions)')
plt.grid(True)

plt.subplot(1, 2, 2)
sns.scatterplot(data=ps4_games_no_outliers, x='user_score_fixed', y='total_sales', alpha=0.5)
plt.title('User Score vs Total Sales (PS4) - Sin Outliers')
plt.xlabel('User Score')
plt.ylabel('Total Sales (millions)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calcular la correlación sin outliers
correlation_critic_no_outliers = ps4_games_no_outliers['critic_score_fixed'].corr(ps4_games_no_outliers['total_sales'])
correlation_user_no_outliers = ps4_games_no_outliers['user_score_fixed'].corr(ps4_games_no_outliers['total_sales'])

print(f"Correlación entre Critic Score y Total Sales (PS4) - Sin Outliers: {correlation_critic_no_outliers:.2f}")
print(f"Correlación entre User Score y Total Sales (PS4) - Sin Outliers: {correlation_user_no_outliers:.2f}")


# *Conclusiones con y sin Outliers
# 
# - Conclusiones con Outliers
# 
# Critic Score vs Total Sales (PS4):
#    Gráfico: Muestra una tendencia general de que los juegos con mejores críticas tienden a vender más, aunque la dispersión es alta.
#    Correlación: 0.33 (relación positiva moderada).
#    Interpretación: Las buenas críticas pueden influir positivamente en las ventas, pero no son el único factor determinante.
#    
# User Score vs Total Sales (PS4):
#    Gráfico: Relación menos clara entre las reseñas de los usuarios y las ventas totales.
#    Correlación: -0.02 (relación muy débil y negativa).
#    Interpretación: Las reseñas de los usuarios no tienen un impacto significativo en las ventas de los juegos para PS4.
#    
# - Conclusiones sin Outliers
# 
# Critic Score vs Total Sales (PS4):
#    Gráfico: La tendencia es menos pronunciada sin los outliers.
#    Correlación: 0.21 (relación positiva débil).
#    Interpretación: Las buenas críticas tienen una influencia más débil en las ventas sin considerar los outliers, indicando que los outliers pueden estar inflando la percepción de la influencia de las críticas.
#    
# User Score vs Total Sales (PS4):
# 
#    Gráfico: Relación aún menos clara entre las reseñas de los usuarios y las ventas totales.
#    Correlación: 0.01 (relación casi inexistente).
#    Interpretación: Sin los outliers, las reseñas de los usuarios tienen una influencia aún menor en las ventas de los juegos para PS4.
# - Conclusión Final
# Dado que los outliers pueden tener un efecto significativo en las percepciones de los datos, es importante considerar ambas perspectivas:
# 
# *Con Outliers:
# Las críticas de los profesionales tienen una influencia moderada en las ventas.
# Las reseñas de los usuarios no tienen un impacto significativo en las ventas.
# *Sin Outliers:
# Las críticas de los profesionales tienen una influencia débil en las ventas.
# Las reseñas de los usuarios tienen una influencia casi inexistente en las ventas.
# 
# Las críticas profesionales pueden influir positivamente en las ventas de los juegos de PS4, pero no son el único factor determinante. Las reseñas de los usuarios tienen muy poca influencia en las ventas, lo que indica que otros factores como el marketing y la reputación del desarrollador o la franquicia juegan un papel más importante en las decisiones de compra de los jugadores.

# Teniendo en cuenta tus conclusiones compara las ventas de los mismos juegos en otras plataformas.

# Filtrar datos para la plataforma PS4
ps4_games = games_relevantes[games_relevantes['platform'] == 'ps4']

# Obtener los nombres de los juegos de PS4 que también están en otras plataformas
ps4_games_names = ps4_games['name'].unique()

# Filtrar datos para los mismos juegos en otras plataformas
other_platforms_games = games_relevantes[games_relevantes['name'].isin(ps4_games_names) & (games_relevantes['platform'] != 'ps4')]

# Combinar datos de PS4 con datos de otras plataformas
combined_data = pd.concat([ps4_games, other_platforms_games])

# Calcular la suma de ventas para cada juego en cada plataforma
ventas_por_juego_y_plataforma = combined_data.groupby(['name', 'platform'])['total_sales'].sum().reset_index()

# Filtrar los 20 juegos más vendidos
top_20_games = ventas_por_juego_y_plataforma.groupby('name')['total_sales'].sum().nlargest(20).index
top_20_data = ventas_por_juego_y_plataforma[ventas_por_juego_y_plataforma['name'].isin(top_20_games)]

# Crear un gráfico de comparación de ventas
plt.figure(figsize=(14, 8))
sns.barplot(x='total_sales', y='name', hue='platform', data=top_20_data, dodge=True)
plt.title('Comparación de ventas de los mismos juegos en diferentes plataformas')
plt.xlabel('Ventas totales (en millones)')
plt.ylabel('Juego')
plt.legend(title='Plataforma')
plt.grid(True)
plt.show()


# El gráfico de barras compara las ventas de los 20 juegos más vendidos en diferentes plataformas.
# 
# Dominancia de PS4 y X360:
# Para muchos de los juegos listados, las plataformas PS4 y X360 tienen ventas significativamente más altas en comparación con otras plataformas. Por ejemplo, "Grand Theft Auto V" y "Call of Duty: Black Ops 3" tienen ventas muy altas en PS4 y X360.
# Esto sugiere que PS4 y X360 son plataformas preferidas por los jugadores para varios juegos populares.
# 
# Ventas en Múltiples Plataformas:
# Juegos como "FIFA 16", "FIFA 17" y "Call of Duty: Advanced Warfare" tienen buenas ventas en múltiples plataformas (PS4, PS3, X360, Xone), lo que indica que estos títulos son populares y tienen una amplia base de jugadores en diferentes plataformas.
# Este fenómeno muestra la importancia de lanzar juegos en múltiples plataformas para maximizar el alcance y las ventas.
# 
# Diferencias en Ventas Entre Plataformas:
# Algunos juegos muestran una clara preferencia de plataforma. Por ejemplo, "The Elder Scrolls V: Skyrim" y "The Last of Us" tienen ventas significativamente más altas en PS4 y PS3 en comparación con otras plataformas.
# Esto puede deberse a la calidad de la experiencia de juego en esas plataformas específicas, la base de usuarios de cada plataforma, y las campañas de marketing dirigidas.
# 
# Ventas Moderadas en Otras Plataformas:
# Juegos como "Diablo III" y "Destiny" tienen ventas moderadas en plataformas como PC y Xone, pero no alcanzan las cifras de ventas en PS4 y X360.
# Esto sugiere que aunque estos juegos son populares, las plataformas PS4 y X360 dominan en términos de ventas totales.
# 
# PS4 y X360: Son las plataformas dominantes para muchos juegos populares. Las estrategias de marketing y desarrollo deben enfocarse en estas plataformas para maximizar las ventas.
# Multiplataforma: Lanzar juegos en múltiples plataformas es una estrategia clave para alcanzar una amplia base de jugadores y maximizar las ventas.
# Optimización y Experiencia: Mejorar la calidad y la experiencia de juego en plataformas populares es crucial para atraer a más jugadores y aumentar las ventas.
# 


# Echa un vistazo a la distribución general de los juegos por género. ¿Qué se puede decir de los géneros más rentables? ¿Puedes generalizar acerca de los géneros con ventas altas y bajas?
# Calcular las ventas totales por género
ventas_por_genero = games_relevantes.groupby('genre')['total_sales'].sum().sort_values(ascending=False).reset_index()

# Mostrar las primeras filas del resultado
print(ventas_por_genero)

# Crear un gráfico de barras para las ventas totales por género
plt.figure(figsize=(14, 8))
sns.barplot(x='total_sales', y='genre', data=ventas_por_genero, palette='viridis')
plt.title('Ventas Totales por Género (2010-2016)')
plt.xlabel('Ventas Totales (en millones)')
plt.ylabel('Género')
plt.grid(True)
plt.show()


# 1. ¿Qué se puede decir de los géneros más rentables?
# Los géneros más rentables son Acción, Disparos, Deportes y Juegos de Rol. Estos géneros dominan las ventas debido a su atractivo amplio y la jugabilidad intensa que ofrecen. Los juegos de acción y disparos son particularmente populares por su dinámica y competencia, mientras que los juegos deportivos atraen a fanáticos leales que buscan simulaciones realistas. Los juegos de rol destacan por sus historias profundas y envolventes.
# 
# 2. ¿Puedes generalizar acerca de los géneros con ventas altas y bajas?
# Géneros con Ventas Altas:
# 
# Acción y Disparos: Estos géneros son los más populares debido a su jugabilidad intensa y competitiva.
# Deportes: La lealtad de los fanáticos a las franquicias deportivas y las actualizaciones anuales impulsan sus altas ventas.
# Juegos de Rol: La profundidad de las historias y la capacidad de personalización atraen a una audiencia dedicada.
# Géneros con Ventas Moderadas a Bajas:
# 
# Moderadas: Misc, Carreras y Plataforma. Estos géneros tienen ventas decentes pero no alcanzan las cifras de los géneros más populares.
# Bajas: Lucha, Simulación, Aventura, Estrategia y Puzzle. Estos géneros tienen audiencias más específicas y limitadas, resultando en ventas más bajas en general.
# Resumen
# Géneros más rentables: Acción, Disparos, Deportes y Juegos de Rol.
# Generalización sobre ventas altas: Géneros de acción, disparos y deportes dominan las ventas debido a su amplia popularidad y jugabilidad atractiva.
# Generalización sobre ventas bajas: Géneros de lucha, simulación, aventura, estrategia y puzzle tienen ventas más bajas debido a su audiencia más específica y limitada.

# Paso 4: Crea un perfil de usuario para cada región

# Calcular las cinco plataformas principales por región
top_5_platforms_na = games_relevantes.groupby('platform')['na_sales'].sum().nlargest(5).reset_index()
top_5_platforms_eu = games_relevantes.groupby('platform')['eu_sales'].sum().nlargest(5).reset_index()
top_5_platforms_jp = games_relevantes.groupby('platform')['jp_sales'].sum().nlargest(5).reset_index()

# Mostrar resultados
print("Top 5 plataformas en NA:")
print(top_5_platforms_na)
print("\nTop 5 plataformas en EU:")
print(top_5_platforms_eu)
print("\nTop 5 plataformas en JP:")
print(top_5_platforms_jp)

# Calcular los cinco géneros principales por región
top_5_genres_na = games_relevantes.groupby('genre')['na_sales'].sum().nlargest(5).reset_index()
top_5_genres_eu = games_relevantes.groupby('genre')['eu_sales'].sum().nlargest(5).reset_index()
top_5_genres_jp = games_relevantes.groupby('genre')['jp_sales'].sum().nlargest(5).reset_index()

# Mostrar resultados
print("Top 5 géneros en NA:")
print(top_5_genres_na)
print("\nTop 5 géneros en EU:")
print(top_5_genres_eu)
print("\nTop 5 géneros en JP:")
print(top_5_genres_jp)

# Comprobar si las clasificaciones de ESRB afectan las ventas en regiones individuales
esrb_sales_na = games_relevantes.groupby('rating_fixed')['na_sales'].sum().reset_index()
esrb_sales_eu = games_relevantes.groupby('rating_fixed')['eu_sales'].sum().reset_index()
esrb_sales_jp = games_relevantes.groupby('rating_fixed')['jp_sales'].sum().reset_index()

# Mostrar resultados
print("Ventas por clasificación ESRB en NA:")
print(esrb_sales_na)
print("\nVentas por clasificación ESRB en EU:")
print(esrb_sales_eu)
print("\nVentas por clasificación ESRB en JP:")
print(esrb_sales_jp)


# Plataformas Líderes: Las plataformas dominantes varían por región, con PS3 y Xbox 360 siendo populares en NA y EU, mientras que Nintendo 3DS y PlayStation Portable son más populares en JP.
# Géneros Populares: Action y Shooter son los géneros más populares en NA y EU, mientras que Role-Playing domina en JP.
# Impacto de ESRB: Las clasificaciones Mature (M) y Everyone (E) dominan las ventas en NA y EU, mientras que en JP los juegos con Rating Pending (RP) tienen una mayor cuota de ventas.


# Paso 5: Prueba de hipótesis

# Hipótesis 1: Xbox One vs. PC
# Filtrar las calificaciones de los usuarios para Xbox One y PC
xone_scores = games_relevantes[games_relevantes['platform'] == 'xone']['user_score_fixed'].dropna()
pc_scores = games_relevantes[games_relevantes['platform'] == 'pc']['user_score_fixed'].dropna()

# Prueba t de Student
t_stat, p_value = stats.ttest_ind(xone_scores, pc_scores, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Decisión basada en el valor p
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios para Xbox One y PC son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula. Las calificaciones promedio de los usuarios para Xbox One y PC son iguales.")


# Hipótesis nula (H₀): No hay diferencia significativa en las calificaciones promedio de los usuarios entre Xbox One y PC.
# Hipótesis alternativa (H₁): Hay una diferencia significativa en las calificaciones promedio de los usuarios entre Xbox One y PC.
# 
# Criterio utilizado:
# 
# Se utilizó la prueba t de Student para dos muestras independientes, ya que queremos comparar las medias de dos grupos diferentes (calificaciones de usuarios para Xbox One y PC).
# Se asumió que las varianzas de las dos poblaciones pueden no ser iguales (prueba t con varianzas desiguales).
# El umbral alfa (
# 𝛼
# α) se estableció en 0.05, que es un valor comúnmente utilizado para determinar significancia estadística.

# Hipótesis 2: Acción vs. Deportes

# Filtrar las calificaciones de los usuarios para los géneros de Acción y Deportes
action_scores = games_relevantes[games_relevantes['genre'] == 'action']['user_score_fixed'].dropna()
sports_scores = games_relevantes[games_relevantes['genre'] == 'sports']['user_score_fixed'].dropna()

# Prueba t de Student
t_stat, p_value = stats.ttest_ind(action_scores, sports_scores, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Decisión basada en el valor p
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula. Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son iguales.")


# Hipótesis nula (H₀): No hay diferencia significativa en las calificaciones promedio de los usuarios entre los géneros de Acción y Deportes.
# Hipótesis alternativa (H₁): Hay una diferencia significativa en las calificaciones promedio de los usuarios entre los géneros de Acción y Deportes.
# 
# Criterio utilizado:
# 
# Se utilizó la prueba t de Student para dos muestras independientes, similar a la primera hipótesis.
# Se asumió que las varianzas de las dos poblaciones pueden no ser iguales (prueba t con varianzas desiguales).
# El umbral alfa (
# 𝛼
# α) se estableció en 0.05.

# Resultados:
# Para la primera hipótesis (Xbox One vs. PC), obtuvimos:
# 
# T-statistic: -1.2851146870744412
# P-value: 0.19926207049502462
# Conclusión: No podemos rechazar la hipótesis nula. Las calificaciones promedio de los usuarios para Xbox One y PC son iguales.
# Para la segunda hipótesis (Acción vs. Deportes), obtuvimos:
# 
# T-statistic: 8.751021923958051
# P-value: 1.1919770523358327e-17
# Conclusión: Rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# 
# - Cómo formulaste las hipótesis nula y alternativa:
# Las hipótesis nula y alternativa se formularon para investigar si existen diferencias significativas en las calificaciones promedio de los usuarios entre dos grupos diferentes (plataformas y géneros). La hipótesis nula siempre plantea que no hay diferencia significativa, mientras que la hipótesis alternativa plantea que sí hay una diferencia.
# 
# - Qué criterio utilizaste para probar las hipótesis y por qué:
# Se utilizó la prueba t de Student para dos muestras independientes con varianzas desiguales. Este método es adecuado para comparar las medias de dos grupos independientes cuando no se puede asumir que las varianzas son iguales. El umbral alfa de 0.05 se utilizó para determinar la significancia estadística, que es un estándar común en análisis estadísticos.


# Hipótesis 1: Xbox One vs. PC sin outliers
# Calcular los cuartiles para identificar y filtrar outliers de Xbox One
Q1_xone = xone_scores.quantile(0.25)
Q3_xone = xone_scores.quantile(0.75)
IQR_xone = Q3_xone - Q1_xone
xone_scores_filtered = xone_scores[~((xone_scores < (Q1_xone - 1.5 * IQR_xone)) |(xone_scores > (Q3_xone + 1.5 * IQR_xone)))]

# Calcular los cuartiles para identificar y filtrar outliers de PC
Q1_pc = pc_scores.quantile(0.25)
Q3_pc = pc_scores.quantile(0.75)
IQR_pc = Q3_pc - Q1_pc
pc_scores_filtered = pc_scores[~((pc_scores < (Q1_pc - 1.5 * IQR_pc)) |(pc_scores > (Q3_pc + 1.5 * IQR_pc)))]

# Prueba t de Student sin outliers
t_stat, p_value = stats.ttest_ind(xone_scores_filtered, pc_scores_filtered, equal_var=False)

print(f"T-statistic sin outliers: {t_stat}")
print(f"P-value sin outliers: {p_value}")

# Decisión basada en el valor p
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios para Xbox One y PC son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula. Las calificaciones promedio de los usuarios para Xbox One y PC son iguales.")

# Hipótesis 2: Acción vs. Deportes sin outliers    
    
# Calcular los cuartiles para identificar y filtrar outliers de Acción
Q1_action = action_scores.quantile(0.25)
Q3_action = action_scores.quantile(0.75)
IQR_action = Q3_action - Q1_action
action_scores_filtered = action_scores[~((action_scores < (Q1_action - 1.5 * IQR_action)) |(action_scores > (Q3_action + 1.5 * IQR_action)))]

# Calcular los cuartiles para identificar y filtrar outliers de Deportes
Q1_sports = sports_scores.quantile(0.25)
Q3_sports = sports_scores.quantile(0.75)
IQR_sports = Q3_sports - Q1_sports
sports_scores_filtered = sports_scores[~((sports_scores < (Q1_sports - 1.5 * IQR_sports)) |(sports_scores > (Q3_sports + 1.5 * IQR_sports)))]

# Prueba t de Student sin outliers
t_stat, p_value = stats.ttest_ind(action_scores_filtered, sports_scores_filtered, equal_var=False)

print(f"T-statistic sin outliers: {t_stat}")
print(f"P-value sin outliers: {p_value}")

# Decisión basada en el valor p
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.")
else:
    print("No podemos rechazar la hipótesis nula. Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son iguales.")


# Conclusión Comparativa
# Influencia de las Reseñas:
# 
# Las reseñas de los críticos tienen una influencia moderada en las ventas cuando se consideran todos los datos (correlación de 0.33), pero esta influencia se reduce ligeramente cuando se eliminan los outliers (correlación de 0.21).
# Las reseñas de los usuarios no muestran una influencia significativa en las ventas en ambos casos, pero la correlación negativa con outliers se vuelve casi nula sin outliers (-0.07 vs 0.01).
# Pruebas de Hipótesis:
# 
# La eliminación de outliers no cambia la conclusión de que no hay diferencias significativas en las calificaciones promedio de los usuarios entre Xbox One y PC.
# Sin embargo, para los géneros de Acción y Deportes, la diferencia en las calificaciones promedio de los usuarios es más pronunciada sin outliers, lo que refuerza la conclusión de que hay una diferencia significativa entre estos géneros.

# Punto 6:
# Conclusión General:
# 

# Este proyecto analizó las ventas de videojuegos en diferentes plataformas y géneros, considerando datos históricos desde 2010 hasta 2016. Los principales hallazgos fueron:
# 
# *Tendencias de Ventas por Plataforma:
# PS3, X360 y PS4 fueron las plataformas con mayores ventas.
# PS4 mostró un crecimiento significativo desde su lanzamiento en 2013 hasta 2016, mientras que PS3 y X360 experimentaron una disminución en ventas.
# 
# *Distribución de Ventas por Género:
# Los géneros más rentables fueron Acción, Disparos y Deportes.
# Los géneros con ventas más bajas incluyeron Estrategia y Puzle.
# 
# *Perfil de Usuario por Región:
# En América del Norte y Europa, las plataformas líderes fueron PS3 y Xbox 360, mientras que en Japón fueron Nintendo 3DS y PlayStation Portable.
# Los géneros más populares en América del Norte y Europa fueron Acción y Disparos, y en Japón, Rol.
# 
# *Impacto de las Reseñas en las Ventas:
# Las reseñas de los críticos mostraron una correlación positiva moderada con las ventas.
# Las reseñas de los usuarios tuvieron una correlación muy débil con las ventas.
# Pruebas de Hipótesis:
# 
# No se encontró una diferencia significativa en las calificaciones promedio de los usuarios entre Xbox One y PC.
# Se encontró una diferencia significativa en las calificaciones promedio entre los géneros de Acción y Deportes.
# - En conclusión:
# En resumen, el análisis reveló que PS4 es una plataforma de gran potencial con un crecimiento continuo en ventas. Los géneros más rentables son Acción, Disparos y Deportes, variando en popularidad según la región. Las reseñas de los críticos tienen un impacto moderado en las ventas, mientras que las reseñas de los usuarios tienen una influencia limitada. Las pruebas de hipótesis confirmaron que las calificaciones de los géneros de Acción y Deportes son significativamente diferentes, mientras que las calificaciones entre Xbox One y PC son similares. Estos hallazgos pueden guiar futuras estrategias de marketing y lanzamiento en la industria de los videojuegos.

