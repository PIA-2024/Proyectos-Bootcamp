
# Paso 4. Análisis exploratorio de datos (Python)

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Cargar los archivos CSV usando las rutas correctas
project_sql_result_01 = pd.read_csv('C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS BOOTCAMP/Analisis exploratorio de datos python-Pruebas de Hipotesis/moved_project_sql_result_01.csv')
project_sql_result_04 = pd.read_csv('C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS BOOTCAMP/Analisis exploratorio de datos python-Pruebas de Hipotesis/moved_project_sql_result_04.csv')
project_sql_result_07 = pd.read_csv('C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS BOOTCAMP/Analisis exploratorio de datos python-Pruebas de Hipotesis/moved_project_sql_result_07.csv')

# Mostrar las primeras filas de cada dataframe para inspeccionar los datos
print(project_sql_result_01.head())
print(project_sql_result_04.head())
print(project_sql_result_07.head())


# In[3]:


# Información del DataFrame project_sql_result_01
print(project_sql_result_01.info())
print()
# Información del DataFrame project_sql_result_04
print(project_sql_result_04.info())
print()
# Información del DataFrame project_sql_result_07
print(project_sql_result_07.info())
print()

# Verificar valores nulos y duplicados en project_sql_result_01
print("Valores nulos en project_sql_result_01:\n", project_sql_result_01.isnull().sum())
print("Valores duplicados en project_sql_result_01:", project_sql_result_01.duplicated().sum())

# Verificar valores nulos y duplicados en project_sql_result_04
print("Valores nulos en project_sql_result_04:\n", project_sql_result_04.isnull().sum())
print("Valores duplicados en project_sql_result_04:", project_sql_result_04.duplicated().sum())

# Verificar valores nulos y duplicados en project_sql_result_07
print("Valores nulos en project_sql_result_07:\n", project_sql_result_07.isnull().sum())

# Asegurarse de que la columna 'start_ts' esté en formato datetime
project_sql_result_07['start_ts'] = pd.to_datetime(project_sql_result_07['start_ts'])

# Identificar los 10 principales barrios en términos de finalización del recorrido
# Ordenar por average_trips y obtener los 10 principales barrios
top_10_neighborhoods = project_sql_result_04.sort_values(by='average_trips', ascending=False).head(10)
print(top_10_neighborhoods)

# Gráfico de empresas de taxis y número de viajes
plt.figure(figsize=(10, 6))
plt.bar(project_sql_result_01['company_name'], project_sql_result_01['trips_amount'])
plt.xticks(rotation=90)
plt.xlabel('Company Name')
plt.ylabel('Number of Trips')
plt.title('Number of Trips by Taxi Company')
plt.show()

# Seleccionar los 10 barrios principales
top_10_neighborhoods = project_sql_result_04.sort_values(by='average_trips', ascending=False).head(10)

# Gráfico de los 10 barrios principales por número de finalizaciones
plt.figure(figsize=(10, 6))
plt.bar(top_10_neighborhoods['dropoff_location_name'], top_10_neighborhoods['average_trips'])
plt.xticks(rotation=90)
plt.xlabel('Neighborhood')
plt.ylabel('Average Trips')
plt.title('Top 10 Neighborhoods by Average Trips')
plt.show()


# Gráfico de empresas de taxis y número de viajes
# Empresas de Taxis y Número de Viajes:
# Las compañías de taxis más populares el 15 y 16 de noviembre de 2017 fueron Flash Cab y Taxi Affiliation Services, con Flash Cab dominando claramente con alrededor de 19,558 viajes.
# Hay una distribución desigual de los viajes entre las diferentes compañías de taxis, con algunas empresas dominando el mercado. Vemos que después de las primeras dos compañías, el número de viajes disminuye considerablemente.
# Varias compañías tienen un número mucho menor de viajes en comparación con las principales.
# Gráfico de los 10 barrios principales por número de finalizaciones
# Top 10 Barrios por Número de Finalizaciones:
# Los barrios con más viajes finalizados en noviembre de 2017 fueron Loop y River North, con Loop teniendo un promedio de aproximadamente 10,727.47 viajes y River North con 9,523.67 viajes.
# Estos barrios son probablemente áreas comerciales y de negocios importantes en Chicago, lo que explica la alta cantidad de viajes que terminan allí.
# Otros barrios en el top 10 incluyen Streeterville, West Loop y O'Hare, que también son áreas de alto tráfico, posiblemente debido a la presencia de oficinas, restaurantes y el aeropuerto.

# # Paso 5: Prueba de hipótesis
# 
# Prueba la hipótesis: "La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare cambia los sábados lluviosos".
# Planteamiento de la hipótesis:
# 
# Hipótesis nula (H0): La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare no cambia los sábados lluviosos.
# Hipótesis alternativa (H1): La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare cambia los sábados lluviosos.
# Decidir el nivel de significación (alfa):
# 
# Nivel de significación: 0.05


# Separar los datos en dos grupos: sábados lluviosos y sábados no lluviosos
rainy_saturdays = project_sql_result_07[(project_sql_result_07['weather_conditions'] == 'Bad') & (project_sql_result_07['start_ts'].dt.weekday == 5)]
non_rainy_saturdays = project_sql_result_07[(project_sql_result_07['weather_conditions'] == 'Good') & (project_sql_result_07['start_ts'].dt.weekday == 5)]

# Calcular la duración promedio para cada grupo
mean_duration_rainy = rainy_saturdays['duration_seconds'].mean()
mean_duration_non_rainy = non_rainy_saturdays['duration_seconds'].mean()

# Prueba t para comparar las medias de los dos grupos
t_stat, p_value = stats.ttest_ind(rainy_saturdays['duration_seconds'], non_rainy_saturdays['duration_seconds'], equal_var=False)

# Mostrar los resultados
print(f"Duración promedio (sábados lluviosos): {mean_duration_rainy} segundos")
print(f"Duración promedio (sábados no lluviosos): {mean_duration_non_rainy} segundos")
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# Decisión basada en el p-value
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula: La duración promedio de los viajes cambia los sábados lluviosos.")
else:
    print("No podemos rechazar la hipótesis nula: La duración promedio de los viajes no cambia los sábados lluviosos.")


# Conclusiones basadas en los resultados del análisis de la hipótesis
# Duración Promedio de los Viajes:
# 
# La duración promedio de los viajes los sábados lluviosos es aproximadamente 2427.21 segundos.
# La duración promedio de los viajes los sábados no lluviosos es aproximadamente 1999.68 segundos.
# Resultados de la Prueba T:
# 
# El valor t obtenido es 7.186, lo que indica una diferencia significativa entre las dos medias.
# El valor p es 6.738994326108734e-12, que es mucho menor que el nivel de significación establecido (0.05).
# Decisión:
# 
# Dado que el valor p es significativamente menor que 0.05, rechazamos la hipótesis nula.
# Esto sugiere que la duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare realmente cambia los sábados lluviosos.
# Explicación de los resultados
# Cómo se plantearon las hipótesis:
# 
# La hipótesis nula (H0) asume que no hay diferencia en la duración promedio de los viajes entre sábados lluviosos y no lluviosos.
# La hipótesis alternativa (H1) asume que hay una diferencia en la duración promedio de los viajes entre sábados lluviosos y no lluviosos.
# Criterio utilizado para probar las hipótesis:
# 
# Se utilizó la prueba t para comparar las medias de dos muestras independientes. Este método es adecuado para comparar las medias de dos grupos diferentes y determinar si la diferencia entre ellas es estadísticamente significativa.
# Conclusiones:
# 
# Los resultados muestran una diferencia significativa en la duración promedio de los viajes entre sábados lluviosos y no lluviosos.
# Los viajes los sábados lluviosos tienden a durar más tiempo en comparación con los sábados no lluviosos, lo cual es consistente con la expectativa de que las condiciones meteorológicas adversas pueden afectar la duración de los viajes.
