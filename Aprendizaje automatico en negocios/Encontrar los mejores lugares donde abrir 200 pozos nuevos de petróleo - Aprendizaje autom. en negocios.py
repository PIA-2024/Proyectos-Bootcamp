# Paso 1: Descarga y preparación de los datos

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos
geo_data_0 = pd.read_csv('geo_data_0.csv')
geo_data_1 = pd.read_csv('geo_data_1.csv')
geo_data_2 = pd.read_csv('geo_data_2.csv')

# Mostrar las primeras filas de cada conjunto de datos
print(geo_data_0.head())
print(geo_data_1.head())
print(geo_data_2.head())


# Información general y descripción de los datos
print(geo_data_0.info())

print(geo_data_1.info())

print(geo_data_2.info())

# Comprobación de valores nulos
print(geo_data_0.isnull().sum())
print(geo_data_1.isnull().sum())
print(geo_data_2.isnull().sum())


# Paso 2: Entrenamiento y Prueba del Modelo
# Vamos a crear un modelo de regresión lineal para predecir el volumen de reservas en pozos nuevos.
# 
# 2.1 División de los Datos
# Dividimos los datos en conjuntos de entrenamiento y validación en una proporción de 75:25.


# Dividir los datos
def split_data(data):
    X = data.drop(['id', 'product'], axis=1)
    y = data['product']
    return train_test_split(X, y, test_size=0.25, random_state=42)

X_train_0, X_val_0, y_train_0, y_val_0 = split_data(geo_data_0)
X_train_1, X_val_1, y_train_1, y_val_1 = split_data(geo_data_1)
X_train_2, X_val_2, y_train_2, y_val_2 = split_data(geo_data_2)

# 2.2 Entrenamiento del Modelo
# Entrenamos un modelo de regresión lineal con los datos de entrenamiento y hacemos predicciones sobre el conjunto de validación.

# Función para entrenar y evaluar el modelo
def train_and_evaluate(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return predictions, rmse

# Entrenar y evaluar modelos para cada región
predictions_0, rmse_0 = train_and_evaluate(X_train_0, y_train_0, X_val_0, y_val_0)
predictions_1, rmse_1 = train_and_evaluate(X_train_1, y_train_1, X_val_1, y_val_1)
predictions_2, rmse_2 = train_and_evaluate(X_train_2, y_train_2, X_val_2, y_val_2)

# Resultados
print("Región 0 - RMSE:", rmse_0)
print("Región 1 - RMSE:", rmse_1)
print("Región 2 - RMSE:", rmse_2)


# 2.3 Guardar Predicciones y Respuestas Correctas
# Guardamos las predicciones y las respuestas correctas para el conjunto de validación.


# Guardar predicciones y respuestas correctas
validation_results_0 = pd.DataFrame({'actual': y_val_0, 'predicted': predictions_0})
validation_results_1 = pd.DataFrame({'actual': y_val_1, 'predicted': predictions_1})
validation_results_2 = pd.DataFrame({'actual': y_val_2, 'predicted': predictions_2})

# Mostrar algunas filas de los resultados
print(validation_results_0.head())
print(validation_results_1.head())
print(validation_results_2.head())


# 2.4 Volumen Medio de Reservas Predicho y RMSE del Modelo
# Calculamos y mostramos el volumen medio de reservas predicho y el RMSE del modelo para cada región.


# Volumen medio de reservas predicho
mean_predictions_0 = predictions_0.mean()
mean_predictions_1 = predictions_1.mean()
mean_predictions_2 = predictions_2.mean()

# Resultados
print("Región 0 - Volumen medio de reservas predicho:", mean_predictions_0)
print("Región 1 - Volumen medio de reservas predicho:", mean_predictions_1)
print("Región 2 - Volumen medio de reservas predicho:", mean_predictions_2)


# 2.5 Análisis de Resultados
# Analizamos los resultados obtenidos de las predicciones y los errores.


# Análisis de resultados
def analyze_results(mean_predictions, rmse, region):
    print(f"Región {region}:")
    print(f" - Volumen medio de reservas predicho: {mean_predictions}")
    print(f" - RMSE del modelo: {rmse}")

analyze_results(mean_predictions_0, rmse_0, 0)
analyze_results(mean_predictions_1, rmse_1, 1)
analyze_results(mean_predictions_2, rmse_2, 2)


# Paso 3: Preparación para el Cálculo de Ganancias

# 3.1 Almacenar valores necesarios para los cálculos


# Almacenar valores necesarios para los cálculos
BUDGET = 100e6  # Presupuesto total en dólares
WELLS = 200  # Número de pozos
REVENUE_PER_UNIT = 4500  # Ingreso por unidad (1000 barriles)
MIN_REVENUE_PER_WELL = BUDGET / WELLS  # Ingreso mínimo por pozo
MIN_UNITS_PER_WELL = MIN_REVENUE_PER_WELL / REVENUE_PER_UNIT  # Unidades mínimas por pozo

print("Cantidad mínima requerida por pozo:", MIN_UNITS_PER_WELL)


# 3.2 Comparar con la cantidad media de reservas en cada región


# Comparación con cantidad media de reservas
mean_reserves_0 = geo_data_0['product'].mean()
mean_reserves_1 = geo_data_1['product'].mean()
mean_reserves_2 = geo_data_2['product'].mean()

print("Región 0 - Cantidad media de reservas:", mean_reserves_0)
print("Región 1 - Cantidad media de reservas:", mean_reserves_1)
print("Región 2 - Cantidad media de reservas:", mean_reserves_2)

# 3.3 Conclusiones sobre cómo preparar el paso para calcular el beneficio
# Para calcular el beneficio, debemos asegurarnos de que los pozos seleccionados cumplan con la cantidad mínima de reservas requerida para evitar pérdidas. Procederemos a seleccionar los pozos con las mayores reservas predichas y calcular el beneficio potencial.

# Paso 4: Cálculo de Ganancias

# 4.1 Elige los 200 pozos con los valores de predicción más altos


# Función para calcular la ganancia
def calculate_profit(predictions, target, count, revenue_per_unit):
    sorted_indices = predictions.argsort()[::-1][:count]
    selected_reserves = target.iloc[sorted_indices]
    total_revenue = selected_reserves.sum() * revenue_per_unit
    return total_revenue - BUDGET

# Calcular ganancias para cada región
profit_0 = calculate_profit(predictions_0, y_val_0, WELLS, REVENUE_PER_UNIT)
profit_1 = calculate_profit(predictions_1, y_val_1, WELLS, REVENUE_PER_UNIT)
profit_2 = calculate_profit(predictions_2, y_val_2, WELLS, REVENUE_PER_UNIT)

print("Región 0 - Ganancia potencial:", profit_0)
print("Región 1 - Ganancia potencial:", profit_1)
print("Región 2 - Ganancia potencial:", profit_2)

# 4.2 Resume el volumen objetivo de reservas y almacena las predicciones

# Convertir las predicciones en Series para usar índices
predictions_0_series = pd.Series(predictions_0)
predictions_1_series = pd.Series(predictions_1)
predictions_2_series = pd.Series(predictions_2)

# Almacenar predicciones para los 200 pozos seleccionados
top_200_predictions_0 = predictions_0_series.nlargest(200)
top_200_predictions_1 = predictions_1_series.nlargest(200)
top_200_predictions_2 = predictions_2_series.nlargest(200)

# Volumen objetivo de reservas
top_200_reserves_0 = y_val_0.iloc[top_200_predictions_0.index].sum()
top_200_reserves_1 = y_val_1.iloc[top_200_predictions_1.index].sum()
top_200_reserves_2 = y_val_2.iloc[top_200_predictions_2.index].sum()

print("Región 0 - Volumen objetivo de reservas (200 pozos):", top_200_reserves_0)
print("Región 1 - Volumen objetivo de reservas (200 pozos):", top_200_reserves_1)
print("Región 2 - Volumen objetivo de reservas (200 pozos):", top_200_reserves_2)


# 4.3 Calcula la ganancia potencial de los 200 pozos principales por región
# Calculado previamente en el paso 4.1.

# 4.4 Presenta tus conclusiones y justifica tu elección
# Basado en las ganancias potenciales calculadas, seleccionamos la región con la ganancia más alta y justificamos nuestra elección.

# Conclusiones
def present_conclusions(profits, reserves, region):
    print(f"Región {region}:")
    print(f" - Ganancia potencial: {profits}")
    print(f" - Volumen objetivo de reservas: {reserves}")

present_conclusions(profit_0, top_200_reserves_0, 0)
present_conclusions(profit_1, top_200_reserves_1, 1)
present_conclusions(profit_2, top_200_reserves_2, 2)

best_region = max([(0, profit_0), (1, profit_1), (2, profit_2)], key=lambda x: x[1])[0]
print(f"La mejor región para el desarrollo de pozos es la región {best_region} debido a su mayor ganancia potencial.")

# Paso 5: Cálculo de Riesgos y Ganancias Utilizando Bootstrapping

# 5.1 Utiliza la técnica de bootstrapping con 1000 muestras


# Función de bootstrapping
def bootstrap_profit(predictions, target, count, revenue_per_unit, n_samples=1000):
    profits = []
    for _ in range(n_samples):
        sample_indices = np.random.choice(predictions.index, size=500, replace=True)
        sample_predictions = predictions[sample_indices]
        sample_target = target.iloc[sample_indices]
        profit = calculate_profit(sample_predictions, sample_target, count, revenue_per_unit)
        profits.append(profit)
    return np.array(profits)

# Realizar bootstrapping para cada región
profits_0 = bootstrap_profit(pd.Series(predictions_0), y_val_0, WELLS, REVENUE_PER_UNIT)
profits_1 = bootstrap_profit(pd.Series(predictions_1), y_val_1, WELLS, REVENUE_PER_UNIT)
profits_2 = bootstrap_profit(pd.Series(predictions_2), y_val_2, WELLS, REVENUE_PER_UNIT)

# 5.2 Encuentra el beneficio promedio, el intervalo de confianza del 95% y el riesgo de pérdidas


# Análisis de resultados de bootstrapping
def analyze_bootstrap_results(profits):
    mean_profit = np.mean(profits)
    lower_bound = np.percentile(profits, 2.5)
    upper_bound = np.percentile(profits, 97.5)
    risk_of_loss = (profits < 0).mean()
    return mean_profit, lower_bound, upper_bound, risk_of_loss

mean_profit_0, lower_bound_0, upper_bound_0, risk_of_loss_0 = analyze_bootstrap_results(profits_0)
mean_profit_1, lower_bound_1, upper_bound_1, risk_of_loss_1 = analyze_bootstrap_results(profits_1)
mean_profit_2, lower_bound_2, upper_bound_2, risk_of_loss_2 = analyze_bootstrap_results(profits_2)

print("Región 0 - Beneficio promedio:", mean_profit_0)
print("Región 0 - Intervalo de confianza 95%:", (lower_bound_0, upper_bound_0))
print("Región 0 - Riesgo de pérdidas:", risk_of_loss_0)

print("Región 1 - Beneficio promedio:", mean_profit_1)
print("Región 1 - Intervalo de confianza 95%:", (lower_bound_1, upper_bound_1))
print("Región 1 - Riesgo de pérdidas:", risk_of_loss_1)

print("Región 2 - Beneficio promedio:", mean_profit_2)
print("Región 2 - Intervalo de confianza 95%:", (lower_bound_2, upper_bound_2))
print("Región 2 - Riesgo de pérdidas:", risk_of_loss_2)


# 5.3 Presenta tus conclusiones y justifica tu elección
# Con base en el análisis de riesgos y ganancias, proponemos la mejor región para el desarrollo de pozos.

# Conclusiones finales
def present_final_conclusions(mean_profit, lower_bound, upper_bound, risk_of_loss, region):
    print(f"Región {region}:")
    print(f" - Beneficio promedio: {mean_profit}")
    print(f" - Intervalo de confianza 95%: ({lower_bound}, {upper_bound})")
    print(f" - Riesgo de pérdidas: {risk_of_loss}")

present_final_conclusions(mean_profit_0, lower_bound_0, upper_bound_0, risk_of_loss_0, 0)
present_final_conclusions(mean_profit_1, lower_bound_1, upper_bound_1, risk_of_loss_1, 1)
present_final_conclusions(mean_profit_2, lower_bound_2, upper_bound_2, risk_of_loss_2, 2)

best_region = max([(0, mean_profit_0), (1, mean_profit_1), (2, mean_profit_2)], key=lambda x: x[1])[0]
print(f"La mejor región para el desarrollo de pozos es la región {best_region} debido a su mayor ganancia potencial.")


# 1. ¿Cuáles son tus hallazgos sobre el estudio de tareas?
# Región 0:
# Tiene el volumen medio de reservas predicho más alto (92.3988) y una RMSE de 37.7566, lo que indica una buena precisión del modelo.
# Región 1:
# Tiene el volumen medio de reservas predicho más bajo (68.7129) y una RMSE extremadamente baja de 0.8903, lo que sugiere que el modelo puede estar sobreajustado o que los datos son muy consistentes.
# Región 2:
# Tiene un volumen medio de reservas predicho (94.7710) similar al de la Región 0, pero con una RMSE de 40.1459, lo que sugiere una mayor variabilidad en los datos de esta región.
# 
# 2. ¿Sugeriste la mejor región para el desarrollo de pozos? ¿Justificaste tu elección?
# 
# Mejor región para el desarrollo de pozos: La Región 1.
# 
# La Región 1 tiene el beneficio promedio más alto después del ajuste utilizando la técnica de bootstrapping correctamente.
# El análisis de bootstrapping mostró que la Región 1 tiene un beneficio promedio de 4,455,171.62 dólares y un intervalo de confianza del 95% que garantiza una mayor estabilidad en los beneficios.
# Además, la Región 1 tiene el riesgo de pérdidas más bajo (1.8%), lo que refuerza la elección debido a la menor incertidumbre asociada.
# 
# Resumen de los resultados:
# 
# Región 0:
# 
# Beneficio promedio: 3,946,406.20 dólares
# Intervalo de confianza 95%: (-1,651,365.00, 9,019,799.16)
# Riesgo de pérdidas: 7%
# 
# Región 1:
# 
# Beneficio promedio: 4,455,171.62 dólares
# Intervalo de confianza 95%: (553,121.07, 8,438,567.11)
# Riesgo de pérdidas: 1.8%
# 
# Región 2:
# 
# Beneficio promedio: 3,751,628.65 dólares
# Intervalo de confianza 95%: (-1,306,189.96, 9,061,252.22)
# Riesgo de pérdidas: 7.5%
# 
# * Conclusión:
# 
# La Región 1 es la mejor opción para el desarrollo de pozos debido a su mayor ganancia potencial promedio y su menor riesgo de pérdidas. La justificación se basa en los siguientes puntos:
# 
# Beneficio promedio más alto: 4,455,171.62 dólares.
# Intervalo de confianza del 95% que muestra estabilidad en los beneficios: (553,121.07, 8,438,567.11).
# Riesgo de pérdidas más bajo: 1.8%.
# Este cambio de conclusión se basa en un análisis más preciso y ajustado utilizando la técnica de bootstrapping correctamente, asegurando que se tomen muestras más representativas y estables.




