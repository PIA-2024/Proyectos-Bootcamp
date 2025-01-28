import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL del sitio web
URL = 'https://practicum-content.s3.us-west-1.amazonaws.com/data-analyst-eng/moved_chicago_weather_2017.html'

# Obtener el contenido de la página
req = requests.get(URL)
req.raise_for_status()  # Asegura que no haya errores de conexión

# Analizar el HTML con BeautifulSoup
soup = BeautifulSoup(req.text, 'lxml')

# Encontrar la tabla con id="weather_records"
table = soup.find('table', attrs={"id": "weather_records"})

if table is None:
    print("No se encontró la tabla con id='weather_records'. Verifica el HTML de la página.")
else:
    # Extraer encabezados de la tabla
    heading_table = [header.text.strip() for header in table.find_all('th')]

    # Extraer contenido de la tabla
    content = []
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if cells:  # Solo procesar filas con datos
            content.append([cell.text.strip() for cell in cells])

    # Crear el DataFrame
    weather_records = pd.DataFrame(content, columns=heading_table)

    # Imprimir el DataFrame completo
    pd.set_option('display.max_rows', None)  # Mostrar todas las filas
    print(weather_records)
